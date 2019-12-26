#include <pybind11/pybind11.h>
namespace py = pybind11;

class pearl {

    public:
        pearl(const std::string &name);
        void setName(const std::string &name_);
        const std::string getName() const;

    private:
        std::string name;

};

PYBIND11_MODULE(pearl, m) {
    m.doc() = "PEARL's implementation in C++"; // module docstring

    py::class_<pearl>(m, "pearl")
        .def(py::init<const std::string &>())
        .def_property("name", &pearl::getName, &pearl::setName)

        .def("__repr__",
            [](const pearl &a) {
                return "<pearl.pearl named '" + a.getName() + "'>";
            }
         );
}
