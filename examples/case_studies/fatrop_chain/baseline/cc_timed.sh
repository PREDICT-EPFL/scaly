#!/bin/sh
# A C compiler that logs its own wall time and peak resident memory to $CC_TIMED_LOG, one line per
# invocation, so the compile cost of CasADi's JIT can be measured from outside CasADi.
log=${CC_TIMED_LOG:-/dev/null}
out=$(mktemp)
start=$(python3 -c 'import time; print(time.time())')
/usr/bin/time -l cc "$@" 2> "$out"
status=$?
end=$(python3 -c 'import time; print(time.time())')
rss=$(awk '/maximum resident set size/ {print $1}' "$out")
grep -v -E "^ +[0-9]+ " "$out" | grep -v "real.*user.*sys" >&2
echo "$start $end ${rss:-0} $status" >> "$log"
rm -f "$out"
exit $status
