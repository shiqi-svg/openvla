# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import time

def get_from_camera():
    return Image.new("RGB", (224, 224))

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0") #读取模型

# Prepare inputs (only once)
image = get_from_camera()
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# ===== 模块级计时工具 =====
class ModuleTimer:
    """用 CUDA events 精确计时各子模块，自动累加多次 forward 调用。"""
    def __init__(self):
        self.timings = {}          # module_name -> list of elapsed_ms
        self._start_events = {}
        self._hooks = []

    def attach(self, model, module_names):
        for name in module_names:
            mod = getattr(model, name)
            self.timings[name] = []
            h1 = mod.register_forward_pre_hook(self._make_pre_hook(name))
            h2 = mod.register_forward_hook(self._make_post_hook(name))
            self._hooks.extend([h1, h2])

    def _make_pre_hook(self, name):
        def hook(module, inp):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._start_events[name] = start
        return hook

    def _make_post_hook(self, name):
        def hook(module, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            elapsed = self._start_events[name].elapsed_time(end)  # ms
            self.timings[name].append(elapsed)
        return hook

    def reset(self):
        for k in self.timings:
            self.timings[k].clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self):
        """返回 {name: total_ms}，total 是该轮 generate 中的累计耗时。"""
        return {name: sum(ts) for name, ts in self.timings.items()}

# ===== Benchmark: 不同 chunk_size 下的推理耗时（含模块分解） =====
ACTION_DIM = 14
CHUNK_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
WARMUP = 3
REPEATS = 10

MODULE_NAMES = ["vision_backbone", "projector", "language_model"]
timer = ModuleTimer()
timer.attach(vla, MODULE_NAMES)

header = (f"{'chunk':>8} {'tokens':>8} {'total(ms)':>11} "
          f"{'vision(ms)':>11} {'proj(ms)':>11} {'LLM(ms)':>11} "
          f"{'vision%':>8} {'proj%':>8} {'LLM%':>8}")
print(header)
print("-" * len(header))

chunks = []
total_infers = []
vision_times = []
proj_times = []
llm_times = []

for chunk_size in CHUNK_SIZES:
    n_tokens = ACTION_DIM * chunk_size

    # Warmup (不计时)
    for _ in range(WARMUP):
        timer.reset()
        with torch.inference_mode():
            vla.generate(inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False,
                         pixel_values=inputs["pixel_values"])

    # Timed runs
    run_total, run_vision, run_proj, run_llm = [], [], [], []
    for _ in range(REPEATS):
        timer.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            vla.generate(inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False,
                         pixel_values=inputs["pixel_values"])
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000

        s = timer.summary()
        run_total.append(wall_ms)
        run_vision.append(s["vision_backbone"])
        run_proj.append(s["projector"])
        run_llm.append(s["language_model"])

    avg = lambda lst: sum(lst) / len(lst)
    t, v, p, l = avg(run_total), avg(run_vision), avg(run_proj), avg(run_llm)

    print(f"{chunk_size:>8} {n_tokens:>8} {t:>11.1f} "
          f"{v:>11.1f} {p:>11.1f} {l:>11.1f} "
          f"{v/t*100:>7.1f}% {p/t*100:>7.1f}% {l/t*100:>7.1f}%")

    chunks.append(chunk_size)
    total_infers.append(round(t, 2))
    vision_times.append(round(v, 2))
    proj_times.append(round(p, 2))
    llm_times.append(round(l, 2))

timer.remove_hooks()

print(f"\nchunk_size = {chunks}")
print(f"Total_infer = {total_infers}")
print(f"Vision      = {vision_times}")
print(f"Projector   = {proj_times}")
print(f"LLM         = {llm_times}")