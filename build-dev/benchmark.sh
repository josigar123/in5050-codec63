#!/bin/bash

N=$1
OUTFILE="tractor-benchmark.txt"

if [ -z "$N" ]; then
    echo "Usage: $0 <num_runs>"
    exit 1
fi

> "$OUTFILE"

for ((i=1; i<=N + 2; i++)); do
    # Capture elapsed time in seconds
    if [ $i -lt 3 ]; then
        echo "Warmup iteration $i"
    else
        echo "Running iteration $i..."
    fi

    t=$({ time ./c63enc -w 1920 -h 1080 -f 690 -o tractor.c63 /mnt/sdcard/cipr/tractor.yuv ; } 2>&1 | grep real | awk '{print $2}')
        # Convert m:ss.s format to seconds
        m=$(echo $t | cut -d"m" -f1)
        s=$(echo $t | cut -d"m" -f2 | sed "s/s//")
        echo "$(echo "$m*60 + $s" | bc -l)" >> "$OUTFILE"
done

