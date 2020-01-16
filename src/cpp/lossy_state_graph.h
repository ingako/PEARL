#ifndef __LOSSY_STATE_GRAPH_H__
#define __LOSSY_STATE_GRAPH_H__

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

using std::cout;
using std::endl;
using std::to_string;
using std::unique_ptr;
using std::make_unique;
using std::unordered_map;
using std::vector;

class lossy_state_graph {
    public:

        lossy_state_graph(int capacity, int window_size);
        int get_next_tree_id(int src);
        void update(int warning_tree_count);

        void try_remove_node(int key);
        void add_node(int key);
        void add_edge(int src, int dest);

        void set_is_stable(bool is_stable_);
        bool get_is_stable();

    private:

        struct node_t {
            int indegree;
            int total_weight;
            unordered_map<int, int> neighbors; // <tree_id, freq>
        };

        vector<unique_ptr<node_t>> graph;
        int capacity;
        int window_size;
        int drifted_tree_counter;
        bool is_stable;
};

#endif
