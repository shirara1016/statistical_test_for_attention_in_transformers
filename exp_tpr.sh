#!/bin/bash

for seed in $(seq 0 9); do
    for method in adaptive bonferroni; do
        for noise_type in iid corr; do
            for data_type in image series; do
                for signal in 1.0 2.0 3.0 4.0; do
                    python experiment/dev.py --method $method --data_type $data_type --noise_type $noise_type --root_seed $seed --signal $signal
                done
            done
        done
    done
done
