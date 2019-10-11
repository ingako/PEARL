#!/usr/bin/env python3

nominal_indices = [0, 2, 3]
dicts = [{}, {}, {}]

with open("airlines-raw.csv", 'r') as f, open("airlines.csv", 'w') as out:
    header = f.readline()
    out.write(header)

    for line in f:
        row = line.split(',')
        for i in range(len(dicts)):
            feature = row[nominal_indices[i]]
            if feature in dicts[i]:
                idx = dicts[i][feature]
            else:
                idx = len(dicts[i])
                dicts[i][feature] = idx
            row[nominal_indices[i]] = idx

        out.write(','.join([str(i) for i in row]))
