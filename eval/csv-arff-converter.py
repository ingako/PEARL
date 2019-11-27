#!/usr/bin/env python3

import csv
import os
import sys

# @relation file
# @attribute a1 numeric
# @attribute class {0,1,2,3,4,5,6,7,8,9}
# @data

base_dir = os.getcwd()
datasets=["elec", "pokerhand", "airlines", "weatherAUS", "covtype"]

for dataset in datasets:
    file_path = f'{base_dir}/{dataset}/{dataset}.csv'
    output_path = f'{dataset}/{dataset}.arff'

    class_labels = set()

    with open(file_path, newline='') as f, open(output_path, 'w') as out:
        reader = csv.reader(f)
        header = next(reader)
        print(header)

        out.write('@relation file\n')

        for idx, h in enumerate(header[:-1]):
            out.write(f'@attribute a{idx} numeric\n')

        for row in reader:
            class_labels.add(row[-1])

        out.write('@attribute class {')
        out.write(','.join(sorted(class_labels)))
        out.write('}\n')

        out.write('@data\n')

    with open(file_path, newline='') as f, open(output_path, 'a') as out:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            out.write(','.join(row))
            out.write('\n')
