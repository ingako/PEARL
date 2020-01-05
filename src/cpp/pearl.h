#ifndef __PEARL_H__
#define __PEARL_H__

#include <deque>
#include <memory>
#include <string>
#include <climits>

#include "code/src/streams/ArffReader.h"
#include "code/src/learners/Classifiers/Trees/HoeffdingTree.h"
#include "code/src/learners/Classifiers/Trees/ADWIN.h"
#include "lru_state.h"

#ifndef NOPYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#define LOG(x) std::cout << (x) << std::endl

using std::string;
using std::unique_ptr;
using std::make_unique;
using std::move;
using std::vector;

class pearl {

    class adaptive_tree {
        public:
            int tree_pool_id;

            adaptive_tree(int tree_pool_id,
                          int kappa_window_size,
                          double warning_delta,
                          double drift_delta);

            void train(Instance& instance);
            int predict(Instance& instance);
            void update_kappa(int actual_labels);
            void reset();

            unique_ptr<HT::HoeffdingTree> tree;
            unique_ptr<adaptive_tree> bg_adaptive_tree;
            unique_ptr<HT::ADWIN> warning_detector;
            unique_ptr<HT::ADWIN> drift_detector;

        private:
            int kappa_window_size;
            double warning_delta;
            double drift_delta;

            double kappa = INT_MIN;
            bool is_candidate = false;

            deque<int> predicted_labels;
    };

    public:
        pearl(int num_trees,
              int repo_size,
              int edit_distance_threshold,
              int kappa_window_size,
              int lossy_window_size,
              int reuse_window_size,
              int arf_max_features,
              double bg_kappa_threshold,
              double cd_kappa_threshold,
              double reuse_rate_upper_bound,
              double warning_delta,
              double drift_delta,
              bool enable_state_adaption);

        void set_num_trees(int num_trees_);
        int get_num_trees() const;

        bool init_data_source(const string& filename);
        bool get_next_instance();
        void prepare_instance(Instance& instance);

        bool process();
        void partial_fit(string instance);

        void select_candidate_trees(vector<char>& target_state,
                                    vector<int>& warning_tree_id_list);

    private:

        int num_trees;
        int repo_size;
        int edit_distance_threshold;
        int kappa_window_size;
        int lossy_window_size;
        int reuse_window_size;
        int num_features;
        int arf_max_features;
        double bg_kappa_threshold;
        double cd_kappa_threshold;
        double reuse_rate_upper_bound;
        double warning_delta;
        double drift_delta;
        bool enable_state_adaption;

        Instance* instance;
        Reader* reader = nullptr;
        vector<unique_ptr<adaptive_tree>> adaptive_trees;
        vector<unique_ptr<adaptive_tree>> candidate_trees;
        vector<unique_ptr<adaptive_tree>> tree_pool;

        bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);
        unique_ptr<adaptive_tree> make_adaptive_tree(int tree_pool_id);

        unique_ptr<lru_state> state_queue;
        vector<char> cur_state;
};

#ifndef NOPYBIND

PYBIND11_MODULE(pearl, m) {
    m.doc() = "PEARL's implementation in C++"; // module docstring

    py::class_<pearl>(m, "pearl")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      double,
                      double,
                      double,
                      double,
                      double,
                      bool>())
        .def_property("num_trees", &pearl::get_num_trees, &pearl::set_num_trees)
        .def("init_data_source", &pearl::init_data_source)
        .def("get_next_instance", &pearl::get_next_instance)
        .def("process", &pearl::process)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has " + std::to_string(p.get_num_trees()) + " trees>";
            }
         );
}

#endif

#endif
