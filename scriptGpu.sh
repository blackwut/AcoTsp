#!/bin/bash

make clean && make acogpu

tspBase="tsp/"

tspArray=(
    "bays29.tsp"
    # "d198.tsp"
    # "pcb442.tsp"
    # "rat783.tsp"
    # "pr1002.tsp"
    # "pcb1173.tsp"
    # "rl1889.tsp"
    # "pr2392.tsp"
    # "fl3795.tsp"
)

nBlocks=(
    1
    2
    4
    8
    16
    32
    64
    128
)

nWarpsPerBlock=(
    1
    2
    4
    8
    16
    32
)


for problem in "${tspArray[@]}"
do
    echo $problem;
    for i in "${nBlocks[@]}"
    do
        for j in "${nWarpsPerBlock[@]}"
        do
            for k in `seq 1 10`;
            do
    #       ./acogpu file.tsp           alpha   beta    q   rho maxEpoch    threadsPerBlock nBlocks nWarpsPerBlock
            ./acogpu $tspBase$problem   1       2       1   0.5 10          128             $i      $j
            done
        done
    done
done
