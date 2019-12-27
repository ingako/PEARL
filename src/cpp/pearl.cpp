#include "pearl.h"

pearl::pearl(int num_trees,
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
             bool enable_state_adaption) :
    num_trees(num_trees),
    repo_size(repo_size),
    edit_distance_threshold(edit_distance_threshold),
    kappa_window(kappa_window),
    lossy_window_size(lossy_window_size),
    reuse_window_size(reuse_window_size),
    arf_max_features(arf_max_features),
    bg_kappa_threshold(bg_kappa_threshold),
    cd_kappa_threshold(cd_kappa_threshold),
    reuse_rate_upper_bound(reuse_rate_upper_bound),
    warning_delta(warning_delta),
    drift_delta(drift_delta),
    enable_state_adaption(enable_state_adaption) {

    pearl::adaptive_tree* at = new pearl::adaptive_tree(1, 1, 1.0, 1.0);

}

void pearl::set_num_trees(int num_trees_) {
    num_trees = num_trees_;
}

int pearl::get_num_trees() const {
    return num_trees;
}

pearl::adaptive_tree::adaptive_tree(int tree_pool_id,
                                    int kappa_window_size,
                                    double warning_delta,
                                    double drift_delta) :
    tree_pool_id(tree_pool_id),
    kappa_window_size(kappa_window_size),
    warning_delta(warning_delta),
    drift_delta(drift_delta) {

    tree = new HT::HoeffdingTree();
    std::cout << "success" << std::endl;
}

void pearl::adaptive_tree::update_kappa(int actual_labels) {

}

void pearl::adaptive_tree::reset() {

}
