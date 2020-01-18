#include "lossy_state_graph.h" 

lossy_state_graph::lossy_state_graph(int capacity,
                                     int window_size) 
        : capacity(capacity), 
          window_size(window_size) {

    is_stable = false;
    graph = vector<unique_ptr<node_t>>(capacity);
}

int lossy_state_graph::get_next_tree_id(int src) {
    if (!graph[src] || graph[src]->total_weight == 0) {
        return -1;
    }

    int r = rand() % graph[src]->total_weight;
    int sum = 0;

    // weighted selection
    for (auto nei : graph[src]->neighbors) {
        sum += nei.second;
        if (r < sum) {
            graph[src]->neighbors[nei.first]++;
            graph[src]->total_weight++;
            return nei.first;
        }
    }

    return -1;
}

bool lossy_state_graph::update(int warning_tree_count) {
    drifted_tree_counter += warning_tree_count;

    if (drifted_tree_counter < window_size) {
        return false;
    }

    drifted_tree_counter -= window_size;

    // lossy count
    for (int i = 0; i < graph.size(); i++) {
        if (!graph[i]) {
            continue;
        }

        for (auto& nei : graph[i]->neighbors) {
            // decrement freq by 1
            graph[i]->total_weight--;
            nei.second--; // freq

            if (nei.second == 0) {
                // remove edge
                graph[nei.first]->indegree--;
                try_remove_node(nei.first);

                graph[i]->neighbors.erase(nei.first);
            }
        }

        try_remove_node(i);
    }

    return true;
}

void lossy_state_graph::try_remove_node(int key) {
    if (graph[key]->indegree == 0 && graph[key]->neighbors.size() == 0) {
        graph[key].reset();
    }
}

void lossy_state_graph::add_node(int key) {
    if (key >= capacity) {
        cout << "id exceeded graph capacity" << endl;
        return;
    }

    graph[key] = make_unique<node_t>();
}

void lossy_state_graph::add_edge(int src, int dest) {
    if (!graph[src]) {
        add_node(src);
    }

    if (!graph[dest]) {
        add_node(dest);
    }

    graph[src]->total_weight++;

    if (graph[src]->neighbors.find(dest) == graph[src]->neighbors.end()) {
        graph[src]->neighbors[dest] = 0;
        graph[dest]->indegree++;
    }

    graph[src]->neighbors[dest]++;
}

void lossy_state_graph::set_is_stable(bool is_stable_) {
    is_stable = is_stable_;
}

bool lossy_state_graph::get_is_stable() {
    return is_stable;
}
