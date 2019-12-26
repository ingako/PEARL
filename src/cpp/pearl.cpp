#include "pearl.h"

pearl::pearl(const std::string &name) : name(name) { }

void pearl::setName(const std::string &name_) {
    name = name_;
}

const std::string pearl::getName() const {
    return name; 
}
