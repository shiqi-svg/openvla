from transformers import AutoModelForVision2Seq
import torch

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print(f"{'Module':<40} {'Params':>15} {'Size (MB)':>12}")
print("-" * 70)

total = 0
for name, child in vla.named_children():
    n = sum(p.numel() for p in child.parameters())
    size_mb = n * 2 / 1024**2  # bfloat16 = 2 bytes
    total += n
    print(f"{name:<40} {n:>15,} {size_mb:>12.1f}")

print("-" * 70)
print(f"{'TOTAL':<40} {total:>15,} {total * 2 / 1024**2:>12.1f}")
