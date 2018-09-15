#!/bin/bash
#Use screen to launch script
# screen -S name                        // create new session
# screen -r name                        // attach to a session

# inside screen
# ./script 1> /dev/null 2> log.txt &
# CTRL + A then D                       // detach from screen session

make clean && make acocpu -j4

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

nThreads=(
    0
    1
    2
    4
    8
    16
    32
    64
    128
)


for problem in "${tspArray[@]}"
do
    echo $problem;
    for i in "${nThreads[@]}"
    do
        for k in `seq 1 10`;
        do
#       ./acocpu file.tsp           alpha   beta    q   rho maxEpoch    mapThreads  farmThreads
            if 
        ./acocpu $tspBase$problem   1       2       1   0.5 10          $i          $i
        done
    done
done



# PARALLEL AND SEQUENTIAL VERSION
# THERE IS THE ISSUE OF OVERLAPPING COUTs

# threadsParallel=(
#     0
#     1
#     2
#     4
#     8
# )

# threadsSequential=(
#     16
#     32
#     64
#     128
# )

# for problem in "${tspArray[@]}"
# do
#     for i in "${threadsParallel[@]}"
#     do
#         for k in `seq 1 10`
#         do
# #       ./acocpu file.tsp           alpha   beta    q   rho maxEpoch    mapThreads  farmThreads
#         ./acocpu $tspBase$problem   1       2       1   0.5 10          $i          $i        1> "/dev/null" 2>> log.txt &
#         pids[${i}]=$!
#         done
#         # wait for all pids
#         for pid in ${pids[*]}; do
#             wait $pid
#         done
#     done

#     for i in "${threadsParallel[@]}"
#     do
#         for k in `seq 1 10`
#         do
# #       ./acocpu file.tsp           alpha   beta    q   rho maxEpoch    mapThreads  farmThreads
#         ./acocpu $tspBase$problem   1       2       1   0.5 10          $i          $i        1> "/dev/null" 2>> log.txt
#         done
#     done
# done
