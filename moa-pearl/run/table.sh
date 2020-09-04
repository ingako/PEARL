#!/usr/bin/env bash

./eval-gain.py $1 $2 $3 # generator seed longest_length
sort -k 1,3 $1/gain-report.txt | column -s',' -t | less

# agrawal-3
# sort -k 1,1 -k 3,3  -k 2,2 -t, $1/gain-report.txt | column -s',' -t | grep -E "999000" | less

# pokerhand
# sort -k 1,1 -k 3,3  -k 2,2 -t, $1/gain-report.txt | column -s',' -t | grep -E "829000" | less
# column -s',' -t $1/gain-report.txt | grep -E "k0.0-e90" | less
