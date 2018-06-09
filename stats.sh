#!/bin/bash
#Creates all the files needed for the report
#./stats.sh ~/Desktop/log_20180608.txt /Volumes/RamDisk/log2.txt
SLOG=$1
DLOG=$2

grep " \*\*\* " $SLOG | wc -l | xargs > $DLOG;
grep " \*\*\* " $SLOG | awk '{print $2,$3,$4,$6,$7,$8,$9}' >> $DLOG


g++ stats.cpp -o stats
#./stats filename tests_same_nThread number_of_different_nThread
./stats $DLOG 10 6
