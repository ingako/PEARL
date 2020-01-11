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
            double kappa = INT_MIN;
            bool is_candidate = false;
            deque<int> predicted_labels;

            adaptive_tree(int tree_pool_id,
                          int kappa_window_size,
                          double warning_delta,
                          double drift_delta);

            void train(Instance& instance);
            int predict(Instance& instance);
            double update_kappa(deque<int> actual_labels, int class_count);
            void reset();

            unique_ptr<HT::HoeffdingTree> tree;
            unique_ptr<adaptive_tree> bg_adaptive_tree;
            unique_ptr<HT::ADWIN> warning_detector;
            unique_ptr<HT::ADWIN> drift_detector;

        private:
            int kappa_window_size;
            double warning_delta;
            double drift_delta;

            double compute_kappa(int* confusion_matrix,
                                 double accuracy,
                                 int sapmle_count,
                                 int class_count);
    };

    public:
        pearl(int num_trees,
              int max_num_candidate_trees,
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

        int get_candidate_tree_group_size() const;
        int get_tree_pool_size() const;

        bool init_data_source(const string& filename);
        bool get_next_instance();
        void prepare_instance(Instance& instance);

        bool process();
        void train(Instance& instance);
        int vote(vector<int> votes);

        void select_candidate_trees(vector<int>& warning_tree_pos_list);

    private:
        struct candidate_tree {
            int tree_pool_id;
            double kappa;
        };

        int num_trees;
        int max_num_candidate_trees;
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

        vector<int> foreground_tree_ids;
        vector<unique_ptr<adaptive_tree>> tree_pool;
        deque<candidate_tree> candidate_trees;

        unique_ptr<lru_state> state_queue;
        vector<char> cur_state;
        deque<int> actual_labels;

        bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);
        unique_ptr<adaptive_tree> make_adaptive_tree(int tree_pool_id);

        void process_basic(vector<int>& votes, int actual_label);
        void process_with_state_adaption(vector<int>& votes, int actual_label);
        void adapt_state(vector<int> drifted_tree_pos_list);
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
                      int,
                      double,
                      double,
                      double,
                      double,
                      double,
                      bool>())
        .def("get_candidate_tree_group_size", &pearl::get_candidate_tree_group_size)
        .def("get_tree_pool_size", &pearl::get_tree_pool_size)
        .def("init_data_source", &pearl::init_data_source)
        .def("get_next_instance", &pearl::get_next_instance)
        .def("process", &pearl::process)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has "
                    + std::to_string(p.get_tree_pool_size()) + " trees>";
            }
         );
}

#endif

#endif
