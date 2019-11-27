#!/usr/bin/env python3

with open("weatherAUS-sorted.csv", 'r') as f, open("weatherAUS.csv", 'w') as out:
    header = f.readline()
    out.write(header)

    dicts = [{} for i in range(len(header.split(',')))]

    for line in f:
        row = line.split(',')

        for i in range(len(row)):
            feature = row[i]

            if feature.isdigit():
                continue

            if feature == 'NA':
                idx = 0

            else:
                if feature in dicts[i]:
                    idx = dicts[i][feature]
                else:
                    idx = len(dicts[i])
                    dicts[i][feature] = idx

            row[i] = idx

        out.write(','.join([str(i) for i in row]))
        out.write('\n')
