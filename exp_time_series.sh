#!/bin/bash

for method in combination fixed; do
    for noise_type in iid corr; do
        for data_size in 128 256 512 1024; do
            python experiment/dev.py --method $method --data_type series --noise_type $noise_type --data_size $data_size
        done
        for architecture_size in small large huge; do
            python experiment/dev.py --method $method --data_type series --noise_type $noise_type --architecture_size $architecture_size
        done
    done
done
