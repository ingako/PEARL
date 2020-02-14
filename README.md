# PEARL

![build](https://github.com/ingako/PEARL/workflows/build/badge.svg)

Implementation of the paper "PEARL: Probabilistic Exact Adaptive Random Forest with Lossy Counting for Data Streams".

> In order to adapt random forests to the dynamic nature of data streams, the state-of-the-art technique discards trained trees and grows new trees when concept drifts are detected. This is particularly wasteful when recurrent patterns exist. In this work, we introduce a novel framework called PEARL, which uses both an exact technique and a probabilistic graphical model with Lossy Counting, to replace drifted trees with relevant trees built from the past. The exact technique utilizes pattern matching to find the set of drifted trees, that co-occurred in predictions in the past. Meanwhile, a probabilistic graphical model is being built to capture the tree replacements among recurrent concept drifts. Once the graphical model becomes stable, it replaces the exact technique and finds relevant trees in a probabilistic fashion. Further, Lossy Counting is applied to the graphical model which brings an added theoretical guarantee on both error rate and space complexity. We empirically show our technique has outperforms baselines in terms of cumulative accuracy on both synthetic and real-world datasets.

### Prerequisites

Python &ge; 3.6

g++ 7.4.0

### Installing

Run

```bash
pip install -r requirements.txt
```

### Options

For a list of available options, run

```bash
python src/main.py -h
```

### Speed Optimization with C++

```bash
cd src/cpp
mkdir build && cd build
cmake ..
make -j8
```

### Example
See `run/run-covtype-cpp.sh` for an example of running PEARL in C++.

Run `run/plot-covtype-demo.py` to plot the results. It should give you results look like the following:

![covtype results](./run/covtype-results.svg)

### Data Preparation

##### Real World Dataset

All the preprocessed real world datasets are under `data/`, including the preprocess scripts to make
the data format compatible with scikit-multiflow. Note 'weatherAUS' is equivalent to 'Rain' in the paper.

The original datasets are available on MOA and Kaggle, see references in the paper.

##### Synthetic Dataset

The scripts for generating abrupt synthetic datasets are availble under
`data/generate-agrawal-[#concepts].py`.

### Reproducing Experiments

The running scripts for PEARL are under `run/`.

The evaluation results have already been included in the according data folders under
`/data/dataset-name]/`.  

##### Real World Dataset

ECPF compatible with [MOA version 2019.05.0](https://github.com/Waikato/moa/tree/2019.05.0) is available [here](https://github.com/ingako/CPF).
The MOA runner script for ECPF is available at `eval/ecpf-runner.sh`.

The evaluation scripts are at `eval/mean-eval-real-world.py` and `eval/gain-eval-real-world.py`.

##### Synthetic Dataset

The evaluation script is at `eval/mean-stat-eval.py`.
