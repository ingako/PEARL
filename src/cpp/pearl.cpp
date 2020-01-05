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

    tree_pool = vector<unique_ptr<adaptive_tree>>(num_trees);

    for (int i = 0; i < num_trees; i++) {
        unique_ptr<adaptive_tree> tree = make_adaptive_tree(i);
        adaptive_trees.push_back(move(tree));
    }

    // initialize LRU state pattern queue
    state_queue = make_unique<lru_state>(100, edit_distance_threshold);

    cur_state = vector<char>(repo_size, '0');
    for (int i = 0; i < num_trees; i++) {
        cur_state[i] = '1';
    }

    state_queue->enqueue(cur_state);
}

unique_ptr<pearl::adaptive_tree> pearl::make_adaptive_tree(int tree_pool_id) {
    return make_unique<adaptive_tree>(tree_pool_id,
                                      kappa_window_size,
                                      warning_delta,
                                      drift_delta);
}

bool pearl::init_data_source(const string& filename) {

    LOG("Initializing data source...");

    reader = new ArffReader();

    if (!reader->setFile(filename)) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    return true;
}

void pearl::prepare_instance(Instance& instance) {
    vector<int> attribute_indices;

    // select random features
    for (int i = 0; i < arf_max_features; i++) {
        attribute_indices.push_back(rand() % num_features);
    }

    instance.setAttributeStatus(attribute_indices);
}

bool pearl::process() {
    int actual_label = instance->getLabel();

    int num_classes = instance->getNumberClasses();
    vector<int> votes(num_classes, 0);

    if (enable_state_adaption) {
        process_with_state_adaption(votes, actual_label);
    } else {
        process_basic(votes, actual_label);
    }

    train(*instance);

    int predicted_label = vote(votes);

    delete instance;

    return predicted_label == actual_label;
}

void pearl::process_with_state_adaption(vector<int>& votes, int actual_label) {

    // keep track of actual labels for candidate tree evaluations
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    int predicted_label;
    vector<char> target_state(cur_state);
    vector<int> warning_tree_id_list;
    vector<int> drifted_tree_pos_list;

    for (int i = 0; i < num_trees; i++) {

        predicted_label = adaptive_trees[i]->predict(*instance);

        votes[predicted_label]++;
        int error_count = (int)(actual_label != predicted_label);

        bool warning_detected_only = false;

        // detect warning
        if (detect_change(error_count, adaptive_trees[i]->warning_detector)) {
            warning_detected_only = false;

            adaptive_trees[i]->bg_adaptive_tree = make_adaptive_tree(-1);
            adaptive_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, adaptive_trees[i]->drift_detector)) {
            warning_detected_only = true;
            drifted_tree_pos_list.push_back(i);
        }

        if (warning_detected_only) {
            warning_tree_id_list.push_back(adaptive_trees[i]->tree_pool_id);
            if (adaptive_trees[i]->tree_pool_id == -1) {
                LOG("Error: tree_pool_id is not updated");
                exit(1);
            }

            target_state[adaptive_trees[i]->tree_pool_id] = '2';
        }
    }

    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->predict(*instance);
    }

    // if warnings are detected, find closest state and update candidate_trees list
    if (warning_tree_id_list.size() > 0) {
        cout << "copy cur_state..." << endl;
        select_candidate_trees(target_state, warning_tree_id_list);
        cout << "copy cur_state completed" << endl;
    }

    // if actual drifts are detected, swap trees and update cur_state
    if (drifted_tree_pos_list.size() > 0) {
        // TODO
        adapt_state(drifted_tree_pos_list);
        state_queue->enqueue(cur_state);
    }
}

void pearl::process_basic(vector<int>& votes, int actual_label) {
    int predicted_label;

    for (int i = 0; i < num_trees; i++) {

        predicted_label = adaptive_trees[i]->predict(*instance);

        votes[predicted_label]++;
        int error_count = (int)(actual_label != predicted_label);

        // detect warning
        if (detect_change(error_count, adaptive_trees[i]->warning_detector)) {
            adaptive_trees[i]->bg_adaptive_tree = make_adaptive_tree(-1);
            adaptive_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, adaptive_trees[i]->drift_detector)) {
            if (adaptive_trees[i]->bg_adaptive_tree) {
                adaptive_trees[i] = move(adaptive_trees[i]->bg_adaptive_tree);
            } else {
                adaptive_trees[i] = make_adaptive_tree(tree_pool.size());
            }
        }
    }
}

int pearl::vote(vector<int> votes) {
    int max_votes = votes[0];
    int predicted_label = 0;

    for (int i = 1; i < votes.size(); i++) {
        if (max_votes < votes[i]) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}

void pearl::train(Instance& instance) {
    for (int i = 0; i < num_trees; i++) {
        // online bagging
        prepare_instance(instance);
        int weight = Utils::poisson(1.0);
        while (weight > 0) {
            weight--;
            adaptive_trees[i]->train(instance);
        }
    }
}

void pearl::select_candidate_trees(vector<char>& target_state,
                                   vector<int>& warning_tree_id_list) {

    vector<char> closest_state = state_queue->get_closest_state(target_state);
    if (closest_state.size() == 0) {
        return;
    }

    for (int i = 0; i < tree_pool.size(); i++) {
        if (cur_state[i] == '0' && closest_state[i] == '1' && tree_pool[i]) {
            // TODO restrict the size of candidate_trees

            candidate_trees.push_back(move(tree_pool[i]));
        }
    }
}

void pearl::adapt_state(vector<int> drifted_tree_pos_list) {
    // sort candiate trees by kappa
    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->update_kappa(actual_labels);
    }
    sort(candidate_trees.begin(), candidate_trees.end(), compare_kappa);

    for (int i = 0; i < drifted_tree_pos_list.size(); i++) {
        // TODO
        if (tree_pool.size() >= repo_size) {
            LOG("early break");
            exit(1);
        }


        int drifted_pos = drifted_tree_pos_list[i];
        unique_ptr<adaptive_tree> drifted_tree = move(adaptive_trees[drifted_pos]);
        unique_ptr<adaptive_tree> swap_tree;

        drifted_tree->update_kappa(actual_labels);

        cur_state[drifted_tree->tree_pool_id] = '0';

        if (candidate_trees.size() > 0
            && candidate_trees.back()->kappa
                - drifted_tree->kappa >= cd_kappa_threshold) {
            swap_tree = move(candidate_trees.back());
            candidate_trees.pop_back();
        }

        if (swap_tree == nullptr) {
            bool add_to_repo = true;

            if (!drifted_tree->bg_adaptive_tree) {
                swap_tree = make_adaptive_tree(-1);

            } else {
                unique_ptr<adaptive_tree> bg_tree =
                    move(drifted_tree->bg_adaptive_tree);
                bg_tree->update_kappa(actual_labels);

                if (bg_tree->kappa == INT_MIN) {
                    // add bg tree to the repo even if it didn't fill the window
                    swap_tree = move(bg_tree);

                } else if (bg_tree->kappa - drifted_tree->kappa >= bg_kappa_threshold) {
                    swap_tree = move(bg_tree);

                } else {
                    // false positive
                    add_to_repo = false;

                }
            }

            if (add_to_repo) {
                swap_tree->reset();

                // assign a new tree_pool_id for background tree
                // and add background tree to tree_pool
                swap_tree->tree_pool_id = tree_pool.size();
                tree_pool.push_back(move(swap_tree));
            }
        }

        cur_state[swap_tree->tree_pool_id] = '1';

        // replace drifted_tree with swap tree
        adaptive_trees[drifted_pos] = move(swap_tree);

        // put drifted tree back to tree_pool
        if (drifted_tree) {
            drifted_tree->reset();
            tree_pool[drifted_tree->tree_pool_id] = move(drifted_tree);
        }
    }
}

bool pearl::detect_change(int error_count,
                          unique_ptr<HT::ADWIN>& detector) {

    double old_error = detector->getEstimation();
    bool error_change = detector->setInput(error_count);

    if (!error_change) {
       return false;
    }

    if (old_error > detector->getEstimation()) {
        // error is decreasing
        return false;
    }

    return true;
}

bool pearl::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();

    num_features = instance->getNumberInputAttributes();
    arf_max_features = log2(num_features) + 1;

    return true;
}

bool pearl::compare_kappa(unique_ptr<adaptive_tree>& tree1,
                          unique_ptr<adaptive_tree>& tree2) {
    return tree1->kappa < tree2->kappa;
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
}

int pearl::adaptive_tree::predict(Instance& instance) {
    double numberClasses = instance.getNumberClasses();
    double* classPredictions = tree->getPrediction(instance);
    int result = 0;
    double max = classPredictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < numberClasses; i++) {
        if (max < classPredictions[i]) {
            max = classPredictions[i];
            result = i;
        }
    }

    if (predicted_labels.size() >= kappa_window_size) {
        predicted_labels.pop_front();
    }
    predicted_labels.push_back(result);

    return result;
}

void pearl::adaptive_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_adaptive_tree) {
        bg_adaptive_tree->train(instance);
    }
}

void pearl::adaptive_tree::update_kappa(deque<int> actual_labels) {

}

void pearl::adaptive_tree::reset() {
    bg_adaptive_tree = nullptr;
    warning_detector->resetChange();
    drift_detector->resetChange();
    predicted_labels.clear();
    kappa = INT_MIN;
}
