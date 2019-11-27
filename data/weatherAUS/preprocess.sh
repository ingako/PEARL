#!/usr/bin/env bash

file=weatherAUS-raw.csv
cat $file | (sed -u 1q; sort --field-separator=',' --key=1) > weatherAUS-sorted.csv

python3 preprocess-weatherAUS.py
