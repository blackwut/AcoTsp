#!/bin/bash
#Use screen to launch script
# screen -S name                        // create new session
# screen -r name                        // attach to a session

# inside screen
# ./script 1> /dev/null 2> log.txt &
# CTRL + A then D                       // detach from screen session

make acocpu

tspBase="tsp/"

tspArray=(
    "bays29.tsp"
    "d198.tsp"
    "pcb442.tsp"
    "rat783.tsp"
    "pr1002.tsp"
    "pcb1173.tsp"
    "rl1889.tsp"
    "pr2392.tsp"
    "fl3795.tsp"
)

mapThreads=(
    1
    2
    4
    8
    16
    32
    64
    128
)

farmThreads=(
    1
    2
    4
    8
    16
    32
    64
)

for problem in "${tspArray[@]}"
do
    echo $problem;
    for i in "${mapThreads[@]}"
    do
        # for j in "${farmThreads[@]}"
        # do
            for k in `seq 1 10`;
            do
    #       ./acocpu file.tsp           alpha   beta    q   rho maxEpoch    mapThreads  farmThreads
            ./acocpu $tspBase$problem   1       2       1   0.5 16           $i          $i
            done
        # done
    done
done


