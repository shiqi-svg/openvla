#!/usr/bin/env python3
"""
OpenVLA Inference Profiler — adjustable chunk_size (n_action_steps) and action_dim.

Usage:
    python benchmark.py --n_action_steps 1 --action_dim 7
    python benchmark.py --n_action_steps 4 --action_dim 14 --num_runs 20
    python benchmark.py --chunk_size 2 --action_dim 7 --no_bf16
"""

import argparse
import statistics
import subprocess
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ── Known device peak TFLOPs (BF16 Tensor Core) ─────────────────────
GPU_PEAK_TFLOPS = {
    "H100": 989.4,
    "H200": 989.4,
    "A100": 312.0,
    "A6000": 155.2,
    "RTX 4090": 330.3,
    "RTX 3090": 142.0,
    "L40": 181.0,
    "L40S": 362.0,
    "RTX PRO 6000": 261.0,
}


def get_device_peak_tflops(device_name: str) -> float:
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in device_name.lower():
            return val
    return 0.0


# ── GPU utilization sampler (background thread) ─────────────────────
class GPUUtilSampler:
    def __init__(self, device=0, interval=0.05):
        self.device = device
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                        f"-i={self.device}",
                    ],
                    timeout=2,
                ).decode().strip()
                self.samples.append(float(out))
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def average(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


# ── CUDA-event forward hook timer ───────────────────────────────────
class HookTimer:
    def __init__(self):
        self._hooks = []
        self._pending = {}
        self.records = {}  # name -> list of (start_event, end_event)

    def attach(self, name: str, module: torch.nn.Module):
        def pre_hook(mod, args, _name=name):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending[_name] = ev

        def post_hook(mod, args, output, _name=name):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            start = self._pending.pop(_name, None)
            if start is not None:
                self.records.setdefault(_name, []).append((start, ev))

        self._hooks.append(module.register_forward_pre_hook(pre_hook))
        self._hooks.append(module.register_forward_hook(post_hook))

    def sync(self) -> dict:
        """Synchronize and return {name: [ms_per_call, ...]}."""
        torch.cuda.synchronize()
        out = {}
        for name, pairs in self.records.items():
            out[name] = [s.elapsed_time(e) for s, e in pairs]
        return out

    def reset(self):
        self.records.clear()
        self._pending.clear()

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Analytical FLOP estimation ──────────────────────────────────────
def estimate_flops(model, prefill_seq_len: int, n_decode_tokens: int) -> dict:
    cfg = model.config.text_config
    H = cfg.hidden_size             # 4096
    I = cfg.intermediate_size       # 11008
    L = cfg.num_hidden_layers       # 32
    V = cfg.vocab_size              # 32064

    # Vision backbone ≈ 2 × params
    vis_params = sum(p.numel() for p in model.vision_backbone.parameters())
    vision_flops = 2 * vis_params

    # Projector MLP (256 patches through 3 linear layers)
    n_patches = 256
    fc1, fc2, fc3 = model.projector.fc1, model.projector.fc2, model.projector.fc3
    proj_flops = 2 * n_patches * (
        fc1.in_features * fc1.out_features
        + fc2.in_features * fc2.out_features
        + fc3.in_features * fc3.out_features
    )

    # LLM prefill
    S = prefill_seq_len
    per_layer_pf = (8 * S * H * H) + (2 * S * S * H) + (6 * S * H * I)
    prefill_flops = L * per_layer_pf + 2 * S * H * V

    # LLM decode (per token, KV cache grows)
    decode_flops = 0
    for i in range(n_decode_tokens):
        kv_len = S + i + 1
        per_tok = L * (8 * H * H + 2 * kv_len * H + 6 * H * I) + 2 * H * V
        decode_flops += per_tok

    video_flops = vision_flops + proj_flops + prefill_flops
    action_flops = decode_flops
    total = video_flops + action_flops

    return dict(
        vision=vision_flops, projector=proj_flops,
        prefill=prefill_flops, decode=decode_flops,
        video=video_flops, action=action_flops, total=total,
    )


# ── Argument parsing ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA Inference Profiler")
    p.add_argument("--n_action_steps", "--chunk_size", type=int, default=1,
                    help="Action chunk size (number of action vectors per inference)")
    p.add_argument("--action_dim", type=int, default=7,
                    help="Action dimension per step (default: 7 for BridgeData V2)")
    p.add_argument("--use_bf16", action="store_true", default=True,
                    help="Use bfloat16 (default)")
    p.add_argument("--no_bf16", action="store_true",
                    help="Use float32 instead of bfloat16")
    p.add_argument("--num_warmup", type=int, default=3,
                    help="Number of warmup iterations")
    p.add_argument("--num_runs", type=int, default=10,
                    help="Number of profiling iterations")
    p.add_argument("--device_peak_tflops", type=float, default=0.0,
                    help="Override device peak BF16 TFLOPs (auto-detected if 0)")
    p.add_argument("--model_path", type=str, default="openvla/openvla-7b",
                    help="HuggingFace model path")
    args = p.parse_args()
    if args.no_bf16:
        args.use_bf16 = False
    return args


# ── Main ────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    n_action_steps = args.n_action_steps
    action_dim = args.action_dim
    total_tokens = n_action_steps * action_dim

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    # ── Prepare inputs ──────────────────────────────────────────────
    image = Image.new("RGB", (224, 224))
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=dtype)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]

    # Token fixup: append empty token 29871 if missing (training format)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)),
            dim=1,
        )
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones(1, 1, device=attention_mask.device, dtype=attention_mask.dtype)),
                dim=1,
            )

    n_text_tokens = input_ids.shape[1]
    n_patches = 256  # (224 / 14)^2
    prefill_seq_len = n_patches + n_text_tokens  # patches inserted after BOS

    # ── Model info ──────────────────────────────────────────────────
    max_action_dim = max(
        len(v["action"]["q01"]) for v in model.norm_stats.values()
    ) if hasattr(model, "norm_stats") and model.norm_stats else action_dim

    latent_params = sum(p.numel() for p in model.projector.parameters())
    action_params = sum(p.numel() for p in model.language_model.model.embed_tokens.parameters())
    num_steps = total_tokens  # autoregressive steps (no denoising)

    # ── FLOP estimate ───────────────────────────────────────────────
    flops = estimate_flops(model, prefill_seq_len, total_tokens)

    # ── Device info ─────────────────────────────────────────────────
    device_name = torch.cuda.get_device_name()
    device_peak_tflops = args.device_peak_tflops
    if device_peak_tflops <= 0:
        device_peak_tflops = get_device_peak_tflops(device_name)

    # ── Attach hooks ────────────────────────────────────────────────
    timer = HookTimer()
    timer.attach("vision_backbone", model.vision_backbone)
    timer.attach("projector", model.projector)
    timer.attach("language_model", model.language_model)
    timer.attach("embed_tokens", model.language_model.model.embed_tokens)

    # ── GPU memory baseline ─────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gpu_mem_before = torch.cuda.memory_allocated() / 1e9

    # ── GPU utilization sampler ─────────────────────────────────────
    gpu_sampler = GPUUtilSampler()

    # ── Warmup ──────────────────────────────────────────────────────
    print(f"Warmup ({args.num_warmup} runs) ...")
    gen_kwargs = dict(
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=total_tokens,
        min_new_tokens=total_tokens,  # force exact token count (no early EOS stop)
        do_sample=False,
    )

    for _ in range(args.num_warmup):
        timer.reset()
        with torch.inference_mode():
            model.generate(input_ids, **gen_kwargs)
        timer.sync()

    # ── Profiling runs ──────────────────────────────────────────────
    all_runs = []
    print(f"Profiling ({args.num_runs} runs) ...")
    torch.cuda.reset_peak_memory_stats()
    gpu_sampler.start()

    for run_idx in range(args.num_runs):
        timer.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            model.generate(input_ids, **gen_kwargs)

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        timings = timer.sync()

        # Vision backbone: 1 call
        vis_ms = sum(timings.get("vision_backbone", [0.0]))
        # Projector: 1 call
        proj_ms = sum(timings.get("projector", [0.0]))
        # Language model: first call = prefill, rest = decode
        llm_calls = timings.get("language_model", [])
        prefill_llm_ms = llm_calls[0] if llm_calls else 0.0
        decode_llm_ms_list = llm_calls[1:] if len(llm_calls) > 1 else []
        decode_llm_ms = sum(decode_llm_ms_list)
        # embed_tokens: first call = text embed (prefill), rest = action token embed (decode)
        embed_calls = timings.get("embed_tokens", [])
        action_embed_ms_list = embed_calls[1:] if len(embed_calls) > 1 else []
        action_embed_ms = sum(action_embed_ms_list)

        video_ms = vis_ms + proj_ms + prefill_llm_ms
        action_ms = decode_llm_ms

        all_runs.append(dict(
            total_infer_ms=total_ms,
            video_ms=video_ms,
            action_ms=action_ms,
            vis_ms=vis_ms,
            proj_ms=proj_ms,
            prefill_llm_ms=prefill_llm_ms,
            decode_llm_ms=decode_llm_ms,
            latent_embed_ms=proj_ms,
            latent_embed_calls=len(timings.get("projector", [])),
            action_embed_ms=action_embed_ms,
            action_embed_calls=len(action_embed_ms_list),
        ))

    gpu_sampler.stop()

    # ── GPU memory ──────────────────────────────────────────────────
    gpu_mem_after = torch.cuda.memory_allocated() / 1e9
    gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9

    # ── Aggregate metrics ───────────────────────────────────────────
    def avg(key):
        return statistics.mean(r[key] for r in all_runs)

    def std(key):
        vals = [r[key] for r in all_runs]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    total_infer_ms = avg("total_infer_ms")
    video_ms = avg("video_ms")
    action_ms = avg("action_ms")

    video_steps = 1
    # Use actual decode call counts from hooks (more reliable than static calculation)
    actual_decode_calls = [r["action_embed_calls"] for r in all_runs]
    action_steps = int(round(statistics.mean(actual_decode_calls))) if actual_decode_calls else max(total_tokens - 1, 0)

    video_pct = video_ms / total_infer_ms * 100 if total_infer_ms > 0 else 0
    action_pct = action_ms / total_infer_ms * 100 if total_infer_ms > 0 else 0

    latent_embed_total = avg("latent_embed_ms")
    latent_embed_calls = int(round(avg("latent_embed_calls")))
    latent_embed_avg = latent_embed_total / latent_embed_calls if latent_embed_calls else 0
    latent_embed_pct = latent_embed_total / total_infer_ms * 100 if total_infer_ms > 0 else 0

    action_embed_total = avg("action_embed_ms")
    action_embed_calls = int(round(avg("action_embed_calls")))
    action_embed_avg = action_embed_total / action_embed_calls if action_embed_calls else 0
    action_embed_pct = action_embed_total / total_infer_ms * 100 if total_infer_ms > 0 else 0

    # ── TFLOPs ──────────────────────────────────────────────────────
    total_tflops = flops["total"] / 1e12
    tflops_per_s_overall = total_tflops / (total_infer_ms / 1000) if total_infer_ms > 0 else 0
    tflops_per_s_video = (flops["video"] / 1e12) / (video_ms / 1000) if video_ms > 0 else 0
    tflops_per_s_action = (flops["action"] / 1e12) / (action_ms / 1000) if action_ms > 0 else 0

    # ── Bandwidth ───────────────────────────────────────────────────
    bytes_per_action = action_dim * 4  # float32 output per action vector
    actions_per_sec = n_action_steps / (total_infer_ms / 1000) if total_infer_ms > 0 else 0
    bandwidth_bytes_per_sec = actions_per_sec * bytes_per_action
    bandwidth_mb_per_sec = bandwidth_bytes_per_sec / 1e6

    # ── GPU utilization ─────────────────────────────────────────────
    avg_gpu_util = gpu_sampler.average

    # ══════════════════════════════════════════════════════════════════
    #  Print results
    # ══════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print(f"  [Profiling] {'=' * 66}")
    print("=" * 80)
    print()
    print(f"  Total _infer          : {total_infer_ms:.1f} ms")
    print(f"  Video  loop        : {video_ms:.1f} ms  ({video_steps} steps), percentage:{video_pct:.1f}%")
    print(f"  Action  loop       : {action_ms:.1f} ms  ({action_steps} steps), percentage:{action_pct:.1f}%")
    print(f"  Latent embed (MLP)    : {latent_embed_total:.1f} ms total / {latent_embed_avg:.2f} ms avg ({latent_embed_calls} calls), percentage:{latent_embed_pct:.1f}%")
    print(f"  Action embed (linear) : {action_embed_total:.1f} ms total / {action_embed_avg:.2f} ms avg ({action_embed_calls} calls), percentage:{action_embed_pct:.1f}%")
    print(f"  TFLOPs total          : {total_tflops:.2f} TFLOPs")
    print(f"  TFLOPs/s overall      : {tflops_per_s_overall:.2f} TFLOPs/s")
    print(f"  TFLOPs/s video        : {tflops_per_s_video:.2f} TFLOPs/s")
    print(f"  TFLOPs/s action       : {tflops_per_s_action:.2f} TFLOPs/s")
    print(f"  Params latent/action  : {latent_params} / {action_params}")
    print(f"  GPU mem before        : {gpu_mem_before:.2f} GB")
    print(f"  GPU mem after         : {gpu_mem_after:.2f} GB")
    print(f"  GPU mem peak          : {gpu_mem_peak:.2f} GB")
    print(f"  GPU utilization       : {avg_gpu_util:.0f} %")
    print()
    print(f"  [Bandwidth] {'=' * 64}")
    print(f"  Action chunk size     : {n_action_steps} steps × {action_dim} dims")
    print(f"  Actions/sec           : {actions_per_sec:.2f} steps/s")
    print(f"  Bandwidth (bytes)     : {bandwidth_bytes_per_sec:.0f} B/s ({bandwidth_mb_per_sec:.4f} MB/s)")
    print(f"  Inference latency     : {total_infer_ms:.1f} ms per chunk")
    print(f"  Control freq (chunk)  : {1000.0 / total_infer_ms:.2f} Hz (new chunk per inference)")
    print(f"  Effective ctrl freq   : {n_action_steps * 1000.0 / total_infer_ms:.2f} Hz (if execute full chunk)")
    print()
    print(f"  [Device Info] {'=' * 63}")
    print(f"  GPU                   : {device_name}")
    print(f"  Device peak TFLOPs    : {device_peak_tflops:.1f} TFLOPs")
    if device_peak_tflops > 0 and device_peak_tflops != float("inf"):
        mfu = tflops_per_s_overall / device_peak_tflops * 100
        print(f"  MFU (overall)         : {mfu:.1f}%")
    print()
    print(f"  [Config] {'=' * 67}")
    print(f"  num_steps (autoregressive) : {num_steps}")
    print(f"  n_action_steps        : {n_action_steps}")
    print(f"  max_action_dim        : {max_action_dim}")
    print(f"  action_dim            : {action_dim}")
    print(f"  dtype                 : {'bfloat16' if args.use_bf16 else 'float32'}")
    print(f"  num_warmup            : {args.num_warmup}")
    print(f"  num_runs              : {args.num_runs}")
    print()
    print(f"  [Timing Std Dev] {'=' * 59}")
    print(f"  Total infer std       : ±{std('total_infer_ms'):.1f} ms")
    print(f"  Video  std         : ±{std('video_ms'):.1f} ms")
    print(f"  Action  std        : ±{std('action_ms'):.1f} ms")
    print()

    timer.detach()


if __name__ == "__main__":
    main()
