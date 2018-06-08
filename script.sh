#!/bin/bash

tspBase="tsp/"

tspArray=(
	"bays29.tsp"
	"berlin52.tsp"
	"d198.tsp"
	"a280.tsp"
	"lin318.tsp"
	"pcb442.tsp"
	"rat783.tsp"
	"pr1002.tsp"
	"nrw1379.tsp"
#"pr2392.tsp"
)

threadsArray=(
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
	for i in "${threadsArray[@]}"
	do
		for j in `seq 1 10`;
		do
#		./acocpu file.tsp alpha beta   q rho maxEpoch nThreads
		./acocpu $tspBase$problem   0.6  0.4 100 0.6       50       $i
		done
	done
done


