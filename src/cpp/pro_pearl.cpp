#include "pro_pearl.h"

pro_pearl::pro_pearl(int num_trees,
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
                     double drift_delta) :
        pearl(num_trees,
              max_num_candidate_trees,
              repo_size,
              edit_distance_threshold,
              kappa_window_size,
              lossy_window_size,
              reuse_window_size,
              arf_max_features,
              bg_kappa_threshold,
              cd_kappa_threshold,
              reuse_rate_upper_bound,
              warning_delta,
              drift_delta,
              true) { }

bool pro_pearl::process() {

    int actual_label = instance->getLabel();

    int num_classes = instance->getNumberClasses();
    vector<int> votes(num_classes, 0);

    pearl::process_with_state_adaption(votes, actual_label);

    if (backtrack_instances.size() >= num_max_backtrack_instances) {
        delete backtrack_instances[0];
        backtrack_instances.pop_front();
    }
    backtrack_instances.push_back(instance);

    pearl::train(*instance);

    int predicted_label = pearl::vote(votes);

    return predicted_label == actual_label;
}

void pro_pearl::select_candidate_trees_proactively() {
    int class_count = instance->getNumberClasses();

    // sort foreground trees by kappa
    for (int i = 0; i < adaptive_trees.size(); i++) {
        adaptive_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(adaptive_trees.begin(), adaptive_trees.end(), compare_kappa);

    // TODO find the worst trees and mark as warning detected
    vector<int> warning_tree_pos_list;
    for (int i = 0; i < 10; i++) {
        warning_tree_pos_list.push_back(i);
    }

    pearl::select_candidate_trees(warning_tree_pos_list);
}

void pro_pearl::adapt_state_proactively() {
    if (candidate_trees.size() == 0) {
        return;
    }

    int class_count = instance->getNumberClasses();

    // sort foreground trees by kappa
    for (int i = 0; i < adaptive_trees.size(); i++) {
        adaptive_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(adaptive_trees.begin(), adaptive_trees.end(), compare_kappa);

    // sort candiate trees by kappa
    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(candidate_trees.begin(), candidate_trees.end(), compare_kappa);

    for (int i = 0; i < adaptive_trees.size(); i++) {
        if (adaptive_trees[i]->kappa < candidate_trees.back()->kappa) {
            adaptive_trees[i].reset();
            candidate_trees.back()->reset();

            adaptive_trees[i] = candidate_trees.back();
            candidate_trees.pop_back();

        } else {
            break;
        }
    }
}

void pro_pearl::adapt_state(vector<int> drifted_tree_pos_list) {

    int class_count = instance->getNumberClasses();

    // sort candiate trees by kappa
    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(candidate_trees.begin(), candidate_trees.end(), compare_kappa);

    shared_ptr<adaptive_tree> first_drifted_tree;
    shared_ptr<adaptive_tree> best_swapped_tree;

    for (int i = 0; i < drifted_tree_pos_list.size(); i++) {
        // TODO
        if (tree_pool.size() >= repo_size) {
            std::cout << "tree_pool full: "
                      << std::to_string(tree_pool.size()) << endl;
            exit(1);
        }

        int drifted_pos = drifted_tree_pos_list[i];
        shared_ptr<adaptive_tree> drifted_tree = adaptive_trees[drifted_pos];
        shared_ptr<adaptive_tree> swap_tree;

        drifted_tree->update_kappa(actual_labels, class_count);

        cur_state[drifted_tree->tree_pool_id] = '0';

        bool add_to_repo = false;

        if (candidate_trees.size() > 0
            && candidate_trees.back()->kappa
                - drifted_tree->kappa >= cd_kappa_threshold) {
            candidate_trees.back()->is_candidate = false;
            swap_tree = candidate_trees.back();
            candidate_trees.pop_back();
        }

        if (swap_tree == nullptr) {
            add_to_repo = true;

            shared_ptr<adaptive_tree> bg_tree = drifted_tree->bg_adaptive_tree;

            if (!bg_tree) {
                swap_tree = make_adaptive_tree(tree_pool.size());

            } else {
                bg_tree->update_kappa(actual_labels, class_count);

                if (bg_tree->kappa == INT_MIN) {
                    // add bg tree to the repo even if it didn't fill the window

                } else if (bg_tree->kappa - drifted_tree->kappa >= bg_kappa_threshold) {

                } else {
                    // false positive
                    add_to_repo = false;

                }

                swap_tree = bg_tree;
            }

            if (add_to_repo) {
                swap_tree->reset();

                // assign a new tree_pool_id for background tree
                // and allocate a slot for background tree in tree_pool
                swap_tree->tree_pool_id = tree_pool.size();
                tree_pool.push_back(swap_tree);

            } else {
                swap_tree->tree_pool_id = drifted_tree->tree_pool_id;

                // TODO
                // swap_tree = move(drifted_tree);
            }
        }

        if (!swap_tree) {
            LOG("swap_tree is nullptr");
            exit(1);
        }

        cur_state[swap_tree->tree_pool_id] = '1';

        // replace drifted_tree with swap tree
        adaptive_trees[drifted_pos] = swap_tree;

        if (!best_swapped_tree) {
            first_drifted_tree = drifted_tree;
            best_swapped_tree = swap_tree;
        }

        drifted_tree->reset();
    }

    backtrack_drifted_trees.push_back(first_drifted_tree);
    backtrack_swapped_trees.push_back(best_swapped_tree);

    state_queue->enqueue(cur_state);
}

int pro_pearl::find_actual_drift_point() {
    if (backtrack_drifted_trees.size() == 0) {
        LOG("No trees to backtrack");
        exit(1);
    }

    shared_ptr<adaptive_tree> drifted_tree = backtrack_drifted_trees[0];
    shared_ptr<adaptive_tree> swapped_tree = backtrack_swapped_trees[0];
    backtrack_drifted_trees.pop_front();
    backtrack_swapped_trees.pop_front();

    int window = 50; // TODO
    int drift_correct = 0;
    int swap_correct = 0;
    double drifted_tree_accuracy = 0.0;
    double swapped_tree_accuracy = 0.0;

    deque<int> drifted_tree_predictions;
    deque<int> swapped_tree_predictions;

    for (int i = backtrack_instances.size() - 1; i >= 0; i--) {
        if (!backtrack_instances[i]) {
            LOG("cur instance is null!");
        }

        int drift_predicted_label = drifted_tree->predict(*backtrack_instances[i], false);
        int swap_predicted_label = swapped_tree->predict(*backtrack_instances[i], false);

        // TODO
        int actual_label = instance->getLabel();

        drift_correct += (int) (drift_predicted_label == actual_label);
        swap_correct += (int) (swap_predicted_label == actual_label);

        if (drifted_tree_predictions.size() >= window) {
            drift_correct -= drifted_tree_predictions[0];
            swap_correct -= swapped_tree_predictions[0];
            drifted_tree_predictions.pop_front();
            swapped_tree_predictions.pop_front();

            drifted_tree_accuracy = (double) drift_correct / window;
            swapped_tree_accuracy = (double) swap_correct / window;

            if (drifted_tree_accuracy >= swapped_tree_accuracy) {
                return backtrack_instances.size() - 1 - i;
            }
        }

    }

    return -1;
}

const bool pro_pearl::get_drift_detected() {
    return drift_detected;
}
