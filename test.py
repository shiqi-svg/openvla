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

# ===== Benchmark: 不同 chunk_size 下的推理耗时 =====
# 原理：chunk_size=K 等效于生成 K*7 个 token，直接用 max_new_tokens=K*7 计时
ACTION_DIM = 7
CHUNK_SIZES = [1, 5, 10, 20, 50]
WARMUP = 2
REPEATS = 5

print(f"{'chunk_size':>12} {'n_tokens':>10} {'avg_time(ms)':>14} {'tokens/sec':>12}")
print("-" * 52)

for chunk_size in CHUNK_SIZES:
    n_tokens = ACTION_DIM * chunk_size
    
    # Warmup
    for _ in range(WARMUP):
        with torch.inference_mode():
            vla.generate(inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False,
                         pixel_values=inputs["pixel_values"])
    
    # Timed runs
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            vla.generate(inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False,
                         pixel_values=inputs["pixel_values"])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    avg_ms = sum(times) / len(times) * 1000
    tps = n_tokens / (avg_ms / 1000)
    print(f"{chunk_size:>12} {n_tokens:>10} {avg_ms:>14.1f} {tps:>12.1f}")
