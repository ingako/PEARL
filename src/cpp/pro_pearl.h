#ifndef __PRO_PEARL_H__
#define __PRO_PEARL_H__

#include "pearl.h"

#ifndef NOPYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


class pro_pearl : public pearl {

    public:

        pro_pearl(int num_trees,
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
                  double drift_delta);

        virtual bool process();
        virtual void adapt_state(vector<int> drifted_tree_pos_list);

        int find_actual_drift_point();
        void select_candidate_trees_proactively();
        void adapt_state_proactively();

    private:

        bool is_proactive = true;
        int num_max_backtrack_instances = 1000;
        deque<Instance*> backtrack_instances;
        deque<shared_ptr<adaptive_tree>> backtrack_drifted_trees;
        deque<shared_ptr<adaptive_tree>> backtrack_swapped_trees;

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
        .def_property_readonly("drift_detected", &pearl::get_drift_detected)
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


    py::class_<pro_pearl, pearl>(m, "pro_pearl")
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
                      double >())
        .def("find_actual_drift_point", &pro_pearl::find_actual_drift_point)
        .def("select_candidate_trees_proactively", &pro_pearl::select_candidate_trees_proactively)
        .def("adapt_state_proactively", &pro_pearl::adapt_state_proactively)
        .def("process", &pro_pearl::process)
        .def("adapt_state", &pro_pearl::adapt_state);
}

#endif

#endif
