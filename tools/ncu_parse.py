"""Parse ncu --page raw output and produce a concise markdown summary.

Usage:
    sudo /usr/local/cuda-12.2/bin/ncu --import <report>.ncu-rep --page raw 2>&1 | python tools/ncu_parse.py
    # or:
    python tools/ncu_parse.py < raw_output.txt
"""
import re
import sys

METRICS = {
    "gpu__time_duration.sum": ("Kernel time", ""),
    "launch__registers_per_thread": ("Registers/thread", ""),
    "launch__shared_mem_per_block_dynamic": ("Dynamic shmem/block", ""),
    "launch__grid_size": ("Grid size", ""),
    "launch__block_size": ("Block size", ""),
    "launch__occupancy_limit_registers": ("Occupancy limit (regs)", "blocks/SM"),
    "launch__occupancy_limit_shared_mem": ("Occupancy limit (shmem)", "blocks/SM"),
    "launch__waves_per_multiprocessor": ("Waves/SM", ""),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("SM throughput", "%"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("Warps active", "%"),
    "sm__maximum_warps_per_active_cycle_pct": ("Occupancy (achieved)", "%"),
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": ("FMA pipe", "%"),
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": ("ALU pipe", "%"),
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed": ("XU/SFU pipe", "%"),
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed": ("LSU pipe", "%"),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("DRAM throughput", "%"),
    "lts__t_sector_hit_rate.pct": ("L2 hit rate", "%"),
    "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed": ("L2 atomic pressure", "%"),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": ("Shmem bank conflicts", ""),
    "l1tex__t_output_wavefronts_pipe_lsu_mem_local_op_ld.sum": ("Local mem loads (spill)", ""),
    "l1tex__t_output_wavefronts_pipe_lsu_mem_local_op_st.sum": ("Local mem stores (spill)", ""),
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum": ("Atomic requests", ""),
    "sm__inst_executed.avg.per_cycle_elapsed": ("IPC", "inst/cycle"),
}

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"

def parse():
    data = {}
    stalls = {}
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("==") or line.startswith("[") or line.startswith("---"):
            continue
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 3:
            metric = parts[0].strip()
            value = parts[-1].strip()
            if metric in METRICS:
                data[metric] = value
            if metric.startswith(STALL_PREFIX) and metric.endswith(STALL_SUFFIX):
                stall_name = metric[len(STALL_PREFIX):-len(STALL_SUFFIX)]
                data[metric] = value
                stalls[stall_name] = value

    print("## ncu Profile Summary\n")
    print("| Metric | Value |")
    print("| --- | --- |")
    for metric_key, (label, unit) in METRICS.items():
        val = data.get(metric_key, "N/A")
        suffix = f" {unit}" if unit and val != "N/A" else ""
        print(f"| {label} | {val}{suffix} |")

    print("\n### Warp Stall Reasons\n")
    print("| Stall | Ratio |")
    print("| --- | --- |")
    sorted_stalls = sorted(stalls.items(), key=lambda x: -float(x[1]) if x[1] != "0" else 0)
    for name, val in sorted_stalls:
        if float(val) >= 0.01:
            print(f"| {name} | {val} |")

if __name__ == "__main__":
    parse()
