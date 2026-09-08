import sys, os
from pathlib import Path
sys.path.insert(0, '/home/ted/.t3/worktrees/alloy/t3code-324f664d')
os.chdir('/home/ted/.t3/worktrees/alloy/t3code-324f664d')
from benchmarks.harness.sweep import run_cell
workload, size, backend = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else 'x'
out = Path(f'/tmp/perf/cells/{workload}_{backend}_{size}_{tag}')
r, info = run_cell(workload, size, backend, out, codegen_timeout=600, compile_timeout=600, max_source_mb=200, benchmark_min_time='0.5s')
rt = float(r['runtime_ns'])/1000 if r.get('runtime_ns') else float('nan')
print(f"{workload} {backend} N={size} [{tag}]  runtime={rt:.2f} us  exec={r['executable_bytes']}  meta={r['static_metadata_bytes']}  w={r['workspace']}  compile={r['compile_status']} run={r['runtime_status']} kcompile={r.get('kernel_compile_ms','')}ms {r.get('note','')}")
