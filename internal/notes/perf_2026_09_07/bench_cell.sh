#!/bin/bash
# usage: bench.sh <celldir> [kernel.c override] [extra cflags...]
set -e
cell=$1; shift
src=${1:-}; shift || true
cd $cell
if [ -z "$src" ]; then src=$(ls *.c | grep -v short | head -1); fi
GB=/home/ted/dev/alloy/benchmarks/third_party/gbench/v1.9.5
clang++ -O3 -std=c++17 -I . -I $GB/include -pthread "$@" -c $src -o kernel_x.o 2>/dev/null
[ -f wrapper.o ] || clang++ -O3 -std=c++17 -I . -I $GB/include -pthread -c benchmark.cpp -o wrapper.o
clang++ -I $GB/include -pthread kernel_x.o wrapper.o -o bench_x $GB/lib/libbenchmark.a -lm
./bench_x --benchmark_min_time=0.5s --benchmark_format=json 2>/dev/null | tail -n +2 | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"benchmarks\"][0][\"cpu_time\"]/1000:.2f} us  ($src $*)')"
