#! /bin/bash

echo "test_n,target_model,%cpu,%mem,time" > resource-benchmark.csv

for i in {1..15};do
    # cnn1d_cpu benchmark
    echo "Launching CNN1D-CPU inference"
    python latency-benchmark.py -t cnn1d_cpu --no-output  &
    PID=$!
    echo "Launching resource monitoring for CNN1D inference"
    while kill -0 $PID 2>/dev/null; do
        echo "$i,cnn1d_cpu,$(ps -p $PID -o %cpu,rss,etime | sed 's/^ *//;s/ *$//;s/ \+/\,/g' | tail -n1 )" >>  resource-benchmark.csv
        sleep 1
    done
    echo "End of LSTM-CPU benchmark"

    # lstm_cpu benchmark
    echo "Launching LSTM-CPU inference"
    python latency-benchmark.py -t lstm_cpu --no-output  &
    PID=$!
    echo "Launching resource monitoring for LSTM inference"
    while kill -0 $PID 2>/dev/null; do
        echo "$i,lstm_cpu,$(ps -p $PID -o %cpu,rss,etime | sed 's/^ *//;s/ *$//;s/ \+/\,/g' | tail -n1 )" >>  resource-benchmark.csv
        sleep 1
    done
    echo "End of LSTM-CPU benchmark"

    # sarima benchmark
    echo "Launching SARIMA inference"
    python latency-benchmark.py -t sarima --no-output  &
    PID=$!
    echo "Launching resource monitoring for SARIMA inference"
    while kill -0 $PID 2>/dev/null; do
        echo "$i,sarima,$(ps -p $PID -o %cpu,rss,etime | sed 's/^ *//;s/ *$//;s/ \+/\,/g' | tail -n1 )" >>  resource-benchmark.csv
        sleep 1
    done
    echo "End of SARIMA benchmark"
done
