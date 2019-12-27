#include <pybind11/pybind11.h>
#include "code/src/learners/Classifiers/Trees/HoeffdingTree.h"

namespace py = pybind11;

using std::string;

class pearl {

    public:
        pearl(int num_trees,
              int repo_size,
              int edit_distance_threshold,
              int kappa_window,
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

        int predict(int X, int y);

    private:

        int num_trees;
        int repo_size;
        int edit_distance_threshold;
        int kappa_window;
        int lossy_window_size;
        int reuse_window_size;
        int arf_max_features;
        double bg_kappa_threshold;
        double cd_kappa_threshold;
        double reuse_rate_upper_bound;
        double warning_delta;
        double drift_delta;
        bool enable_state_adaption;

    class adaptive_tree {
        public:
            adaptive_tree(int tree_pool_id,
                          int kappa_window_size,
                          double warning_delta,
                          double drift_delta);

            void update_kappa(int actual_labels);
            void reset();

         private:
             HT::HoeffdingTree* tree;
             int tree_pool_id;
             int kappa_window_size;
             double warning_delta;
             double drift_delta;

    };

};

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
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has " + std::to_string(p.get_num_trees()) + " trees>";
            }
         );
}
