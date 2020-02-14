cmake_minimum_required(VERSION 3.10)

project(pearl VERSION 0.0.1)

set(sourcefiles
streamDM/src/core/Attribute.cpp
streamDM/src/core/DenseInstance.cpp
streamDM/src/core/DiscreteEstimator.cpp
streamDM/src/core/DoubleVector.cpp
streamDM/src/core/GaussianEstimator.cpp
streamDM/src/core/Instance.cpp
streamDM/src/core/InstanceInformation.cpp
streamDM/src/core/SparseEstimator.cpp
streamDM/src/core/SparseInstance.cpp
streamDM/src/core/SplitCriterion.cpp
streamDM/src/utils/Utils.cpp
streamDM/src/utils/CommandLineParser.cpp
streamDM/src/utils/Configurable.cpp
streamDM/src/utils/DynamicCreateClass.cpp
streamDM/src/utils/jsoncpp.cpp
streamDM/src/utils/LearnerModel.cpp
streamDM/src/utils/RTTI.cpp
streamDM/src/streams/Reader.cpp
streamDM/src/streams/ArffReader.cpp
streamDM/src/streams/C45Reader.cpp
streamDM/src/streams/CSVReader.cpp
streamDM/src/streams/LibSVMReader.cpp
streamDM/src/evaluation/BasicClassificationEvaluator.cpp
streamDM/src/evaluation/Evaluator.cpp
streamDM/src/evaluation/Measures.cpp
streamDM/src/evaluation/NullEvaluator.cpp
streamDM/src/learners/Learner.cpp
streamDM/src/learners/Classifiers/Functions/Logisticregression.cpp
streamDM/src/learners/Classifiers/Functions/Majorityclass.cpp
streamDM/src/learners/Classifiers/Functions/Perceptron.cpp
streamDM/src/learners/Classifiers/Bayes/NaiveBayesStatistics.cpp
streamDM/src/learners/Classifiers/Bayes/Naivebayes.cpp
streamDM/src/learners/Classifiers/Bayes/SimpleNaiveBayesStatistics.cpp
streamDM/src/learners/Classifiers/Meta/Bagging.cpp
streamDM/src/learners/Classifiers/Bayes/observer/NominalAttributeClassObserver.cpp
streamDM/src/learners/Classifiers/Bayes/observer/NumericAttributeClassObserver.cpp
streamDM/src/learners/Classifiers/Trees/ADWIN.cpp
streamDM/src/learners/Classifiers/Trees/AttributeSplitSuggestion.cpp
streamDM/src/learners/Classifiers/Trees/HATNode.cpp
streamDM/src/learners/Classifiers/Trees/HoeffdingAdaptiveTree.cpp
streamDM/src/learners/Classifiers/Trees/HoeffdingTree.cpp
streamDM/src/learners/Classifiers/Trees/HTAttributeClassObserver.cpp
streamDM/src/learners/Classifiers/Trees/HTNode.cpp
streamDM/src/learners/Classifiers/Trees/InstanceConditionalTest.cpp
streamDM/src/tasks/EvaluatePrequential.cpp
streamDM/src/tasks/EvaluateHoldOut.cpp
streamDM/src/tasks/Task.cpp)

include_directories(
streamDM/src/core
streamDM/src/learners
streamDM/src
streamDM/src/utils
streamDM/src/learners/classifiers/Bayes
streamDM/src/tasks
streamDM/src/learners/classifiers/Trees
streamDM/src/learners/Classifiers/Bayes/observer/
streamDM/src/learners/Classifiers/Meta/
streamDM/src/learners/Classifiers/Functions/
streamDM/src/evaluation/
streamDM/src/streams/
spdlog/include
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_library(streamDM STATIC ${sourcefiles})

add_subdirectory(pybind11)
pybind11_add_module(pearl pearl_bindings.cpp pearl.cpp pro_pearl.cpp lru_state.cpp lossy_state_graph.cpp)

add_subdirectory(spdlog)
spdlog_enable_warnings(pearl)

target_link_libraries(pearl PRIVATE streamDM spdlog::spdlog)
