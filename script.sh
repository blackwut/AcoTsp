#!/bin/bash

tspBase="tsp/"

tspArray=(
	"bays29.tsp"
	"berlin52.tsp"
	"d198.tsp"
	"d493.tsp"
	"d657.tsp"
)

threadsArray=(
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
	for i in "${threadsArray[@]}"
	do
#		./acocpu file.tsp alpha beta   q rho maxEpoch nThreads
		./acocpu $tspBase$problem   0.6  0.4 100 0.6       50       $i
	done
done

