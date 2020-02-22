#!/usr/bin/env python

import pandas as pd
import shutil

id_to_class = {}
class_to_code = {"banana":0, "wine":1, "background":2}

with open("HT_Sensor_metadata.dat") as f:
    header = f.readline()

    for line in f:
        row = line.split()
        id_to_class[int(row[0])] = class_to_code[row[2]]

df = pd.read_table("HT_Sensor_dataset.dat", sep="\s+")

df['id'] = df['id'].map(lambda x: id_to_class[x])
df = df.sort_values('time')

# swap cols
cols = list(df.columns)

first_col = cols[0]
last_col = cols[len(cols) - 1]
cols[0], cols[len(cols) - 1] = last_col, first_col

df=df.reindex(columns=cols)

df.to_csv("gas.csv", sep=',', index=None, header=False)

# generate arff headers
with open("headers.txt", "w") as out:
    for col in cols[:-1]:
        out.write(f"@attribute {col} numeric\n")
    out.write("@attribute class {0,1,2}\n")
    out.write("\n@data")

# merge arff headers and data files
with open('gas.arff','wb') as wfd:
    for f in ["headers.txt", "gas.csv"]:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
