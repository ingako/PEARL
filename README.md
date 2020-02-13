# PEARL

Probabilistic and Exact Adaptive Random Forest with Lossy Counting

### Prerequisites

Python &ge; 3.6

### Installing

Run

```
pip install -r requirements.txt
```

### Options

For a list of available options, run

```
python src/main.py -h
```

### Speed Optimization with C++

Make sure [log4cpp](http://log4cpp.sourceforge.net/) has been installed.

```
cd src/cpp
mkdir build
cd build
cmake ..
make
```

### Example
See `run/run-covtype-cpp.sh` for an example of running PEARL in C++.

Run `run/plot-covtype-demo.py` to plot the results. It should give you results look like the following:

[covtype results](https://github.com/ingako/PEARL/run/covtype-results.png)

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
