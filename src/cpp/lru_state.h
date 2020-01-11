#ifndef __LRU_STATE_H__
#define __LRU_STATE_H__

#include <climits>
#include <unordered_map>
#include <list>
#include <vector>

using std::string;
using std::vector;
using std::unordered_map;
using std::list;

class lru_state {
    public:
        lru_state(int capacity, int distance_threshold);

        vector<char> get_closest_state(vector<char> target_pattern);
        string to_string();
        void update_queue(vector<char> pattern);
        void enqueue(vector<char> pattern);

    private:
        struct state {
            vector<char> pattern;
            int val;
            int freq;
            state(vector<char> pattern, int val, int freq) : pattern(pattern),
            val(val), freq(freq) {}
        };

        list<state> queue;
        unordered_map<string, list<state>::iterator> map;
        int capacity;
        int distance_threshold; 
};

#endif
