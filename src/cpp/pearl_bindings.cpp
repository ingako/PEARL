#include <pybind11/pybind11.h>
#include "pearl.h"
#include "pro_pearl.h"

namespace py = pybind11;

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
                      bool,
                      bool>())
        .def("get_candidate_tree_group_size", &pearl::get_candidate_tree_group_size)
        .def("get_tree_pool_size", &pearl::get_tree_pool_size)
        .def("init_data_source", &pearl::init_data_source)
        .def("get_next_instance", &pearl::get_next_instance)
        .def("process", &pearl::process)
        .def("is_state_graph_stable", &pearl::is_state_graph_stable)
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
        .def_property_readonly("drift_detected", &pro_pearl::get_drift_detected)
        .def("find_last_actual_drift_point", &pro_pearl::find_last_actual_drift_point)
        .def("select_candidate_trees_proactively", &pro_pearl::select_candidate_trees_proactively)
        .def("adapt_state_proactively", &pro_pearl::adapt_state_proactively)
        .def("process", &pro_pearl::process)
        .def("adapt_state", &pro_pearl::adapt_state);
}
