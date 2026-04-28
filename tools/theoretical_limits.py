#!/usr/bin/env python3
"""Compute theoretical performance limits for 30M scat/s target."""
import sys
sys.stdout.reconfigure(line_buffering=True)

total_warp_inst = 5.46e9
ipc_current = 2.45
ipc_max = 4.0
n_sm = 128
clock_ghz = 2.52

warp_rate = ipc_current * n_sm * clock_ghz * 1e9
time_current = total_warp_inst / warp_rate
print(f"Current: {time_current*1e3:.2f}ms (IPC={ipc_current})")

warp_rate_max = ipc_max * n_sm * clock_ghz * 1e9
time_max_ipc = total_warp_inst / warp_rate_max
print(f"Max IPC ({ipc_max}): {time_max_ipc*1e3:.2f}ms = {100000/(time_max_ipc)/1e6:.1f}M scat/s")

fp32_frac = 0.71
inst_fp16 = total_warp_inst * (1 - fp32_frac) + total_warp_inst * fp32_frac / 2
for ipc in [2.45, 3.0, 3.5, 4.0]:
    wr = ipc * n_sm * clock_ghz * 1e9
    t = inst_fp16 / wr
    print(f"fp16 half2 + IPC={ipc}: {t*1e3:.2f}ms = {100000/t/1e6:.1f}M scat/s")

for ipc in [3.0, 3.5, 3.8]:
    wr = ipc * n_sm * clock_ghz * 1e9
    t = total_warp_inst / wr
    print(f"Pipeline IPC={ipc}: {t*1e3:.2f}ms = {100000/t/1e6:.1f}M scat/s")

print(f"\nTarget: 3.33ms = 30.0M scat/s")
print(f"Compute floor at max IPC: {time_max_ipc*1e3:.2f}ms = {100000/time_max_ipc/1e6:.1f}M scat/s")
