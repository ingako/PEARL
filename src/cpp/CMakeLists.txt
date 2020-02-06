cmake_minimum_required(VERSION 3.10)

project(pearl VERSION 0.0.1)

set(sourcefiles
code/src/core/Attribute.cpp
code/src/core/DenseInstance.cpp
code/src/core/DiscreteEstimator.cpp
code/src/core/DoubleVector.cpp
code/src/core/GaussianEstimator.cpp
code/src/core/Instance.cpp
code/src/core/InstanceInformation.cpp
code/src/core/SparseEstimator.cpp
code/src/core/SparseInstance.cpp
code/src/core/SplitCriterion.cpp
code/src/utils/Utils.cpp
code/src/utils/Log.cpp
code/src/utils/CommandLineParser.cpp
code/src/utils/Configurable.cpp
code/src/utils/DynamicCreateClass.cpp
code/src/utils/jsoncpp.cpp
code/src/utils/LearnerModel.cpp
code/src/utils/RTTI.cpp
code/src/streams/Reader.cpp
code/src/streams/ArffReader.cpp
code/src/streams/C45Reader.cpp
code/src/streams/CSVReader.cpp
code/src/streams/LibSVMReader.cpp
code/src/evaluation/BasicClassificationEvaluator.cpp
code/src/evaluation/Evaluator.cpp
code/src/evaluation/Measures.cpp
code/src/evaluation/NullEvaluator.cpp
code/src/learners/Learner.cpp
code/src/learners/Classifiers/Functions/Logisticregression.cpp
code/src/learners/Classifiers/Functions/Majorityclass.cpp
code/src/learners/Classifiers/Functions/Perceptron.cpp
code/src/learners/Classifiers/Bayes/NaiveBayesStatistics.cpp
code/src/learners/Classifiers/Bayes/Naivebayes.cpp
code/src/learners/Classifiers/Bayes/SimpleNaiveBayesStatistics.cpp
code/src/learners/Classifiers/Meta/Bagging.cpp
code/src/learners/Classifiers/Bayes/observer/NominalAttributeClassObserver.cpp
code/src/learners/Classifiers/Bayes/observer/NumericAttributeClassObserver.cpp
code/src/learners/Classifiers/Trees/ADWIN.cpp
code/src/learners/Classifiers/Trees/AttributeSplitSuggestion.cpp
code/src/learners/Classifiers/Trees/HATNode.cpp
code/src/learners/Classifiers/Trees/HoeffdingAdaptiveTree.cpp
code/src/learners/Classifiers/Trees/HoeffdingTree.cpp
code/src/learners/Classifiers/Trees/HTAttributeClassObserver.cpp
code/src/learners/Classifiers/Trees/HTNode.cpp
code/src/learners/Classifiers/Trees/InstanceConditionalTest.cpp
code/src/tasks/EvaluatePrequential.cpp
code/src/tasks/EvaluateHoldOut.cpp
code/src/tasks/Task.cpp)

include_directories(
code/src/core
code/src/learners
code/src
code/src/utils
code/src/learners/classifiers/Bayes
code/src/tasks
code/src/learners/classifiers/Trees
code/src/learners/Classifiers/Bayes/observer/
code/src/learners/Classifiers/Meta/
code/src/learners/Classifiers/Functions/
code/src/evaluation/
code/src/streams/
/usr/local/include/
)

# log4cpp
find_library(LOG4CPP_LIBRARY log4cpp)
link_directories(/usr/local/lib/)
find_package(Threads REQUIRED)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_library(streamDM STATIC ${sourcefiles})

add_subdirectory(pybind11)
pybind11_add_module(pearl pearl_bindings.cpp pearl.cpp pro_pearl.cpp lru_state.cpp lossy_state_graph.cpp)

target_link_libraries(pearl PRIVATE streamDM ${LOG4CPP_LIBRARY})
