#ifndef __PRO_PEARL_H__
#define __PRO_PEARL_H__

#include "pearl.h"

#ifndef NOPYBIND
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


#endif

#endif
