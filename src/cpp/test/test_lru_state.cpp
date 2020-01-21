#include <assert.h>
#include <iostream>
#include "lru_state.h"

int main() {
    lru_state* state_queue = new lru_state(4, 0);


    set<int> n0 = {0};
    set<int> n1 = {4};
    set<int> n2 = {3};
    set<int> n3 = {3, 4};
    set<int> n4 = {2};
    
    vector<set<int>> states = {n0, n1, n2, n0,n3, n4};
    
    vector<string> expected_result = {
        "0:1->",
        "4:1->0:1->",
        "3:1->4:1->0:1->",
        "0:2->3:1->4:1->",
        "3,4:1->0:2->3:1->4:1->",
        "2:1->3,4:1->0:2->3:1->"
    };
    
    set<int> ids_to_exclude;

    for (int i = 0; i < states.size(); i++) {
        set<int> closest_state = state_queue->get_closest_state(states[i], ids_to_exclude);
        state_queue->enqueue(states[i]);

        string cur_state_queue = state_queue->to_string();
        // std::cout << cur_state_queue << std::endl;
        assert(cur_state_queue == expected_result[i]);
    }
    
    return 0;
}
