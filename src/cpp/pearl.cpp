#include "pearl.h"

pearl::pearl(int num_trees,
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
             bool enable_state_adaption) :
    num_trees(num_trees),
    repo_size(repo_size),
    edit_distance_threshold(edit_distance_threshold),
    kappa_window_size(kappa_window_size),
    lossy_window_size(lossy_window_size),
    reuse_window_size(reuse_window_size),
    arf_max_features(arf_max_features),
    bg_kappa_threshold(bg_kappa_threshold),
    cd_kappa_threshold(cd_kappa_threshold),
    reuse_rate_upper_bound(reuse_rate_upper_bound),
    warning_delta(warning_delta),
    drift_delta(drift_delta),
    enable_state_adaption(enable_state_adaption) {

    for (int i = 0; i < num_trees; i++) {
        adaptive_tree* tree = new adaptive_tree(i,
                                                kappa_window_size,
                                                warning_delta,
                                                drift_delta);
        adaptive_trees.push_back(tree);
    }
}

bool pearl::init_data_source(const string& filename) {

    std::cout << "Initializing data source..." << std::endl;

    reader = new ArffReader();

    if (!reader->setFile(filename)) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    return true;
}

bool pearl::process() {
    int predicted_label = this->predict(*instance);
    int actual_label = instance->getLabel();

    int num_classes = instance->getNumberClasses();

    if (actual_label != predicted_label) {
        LOG("wrong prediction");
    } else {
        LOG("correct prediction");
    }

    adaptive_trees[0]->tree->train(*instance);

    return actual_label == predicted_label;
}

int pearl::predict(Instance& instance) {
    double numberClasses = instance.getNumberClasses();
    double* classPredictions = adaptive_trees[0]->tree->getPrediction(instance);
    int result = 0;
    double max = classPredictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < numberClasses; i++) {
        if (max < classPredictions[i]) {
            max = classPredictions[i];
            result = i;
        }
    }

    return result;
}

bool pearl::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();
    return true;
}

void pearl::set_num_trees(int num_trees_) {
    num_trees = num_trees_;
}

int pearl::get_num_trees() const {
    return num_trees;
}

// class adaptive_tree
pearl::adaptive_tree::adaptive_tree(int tree_pool_id,
                                    int kappa_window_size,
                                    double warning_delta,
                                    double drift_delta) :
    tree_pool_id(tree_pool_id),
    kappa_window_size(kappa_window_size),
    warning_delta(warning_delta),
    drift_delta(drift_delta) {

    tree = make_unique<HT::HoeffdingTree>();
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);

    std::cout << "init an adaptive tree" << std::endl;
}

void pearl::adaptive_tree::update_kappa(int actual_labels) {

}

void pearl::adaptive_tree::reset() {

}
