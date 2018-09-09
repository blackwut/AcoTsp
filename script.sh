#!/bin/bash
#To lunch script in background and exiting from ssh to avoid termination
#nohup ./script &> log.txt &

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
		for j in "${farmThreads[@]}"
		do
			for k in `seq 1 10`;
			do
	#		./acocpu file.tsp			alpha	beta   	q	rho	maxEpoch 	mapThreads	farmThreads
			./acocpu $tspBase$problem	1		2		1	0.5	25			$i			$j
			done
		done
	done
done


