#include "lru_state.h"

lru_state::lru_state(int capacity, int distance_threshold)
    : capacity(capacity), distance_threshold(distance_threshold) {}

vector<char> lru_state::get_closest_state(vector<char> target_pattern) {
    string target_key(std::begin(target_pattern), std::end(target_pattern));

    int min_edit_distance = INT_MAX;
    int max_freq = 0;
    vector<char> closest_pattern;

    // find the smallest edit distance
    for (auto cur_state : queue) {
        vector<char> cur_pattern = cur_state.pattern;
        
        int cur_freq = cur_state.freq;
        int cur_edit_distance = 0;

        bool update_flag = true;
        for (int i = 0; i < target_pattern.size(); i++) {
            if (cur_pattern[i] == target_pattern[i]) {
                continue;
            }

            // tree with drift must be unset
            if (cur_pattern[i] == '1' && target_pattern[i] == '2') {
                update_flag = false;
                break;
            }

            cur_edit_distance++;

            if (cur_edit_distance > distance_threshold
                    || cur_edit_distance > min_edit_distance) {
                update_flag = false;
                break;
            }
        }

        if (!update_flag) {
            continue;
        }
        
        if (min_edit_distance == cur_edit_distance && cur_freq < max_freq) {
            continue;
        }

        min_edit_distance = cur_edit_distance;
        max_freq = cur_freq;
        closest_pattern = cur_pattern;
    }

    return closest_pattern;
}

string lru_state::to_string() {
    string s = "";

    list<state>::iterator it;
    for (it = queue.begin(); it != queue.end(); it++) {
        vector<char> cur_pattern = it->pattern;
        string freq = std::to_string(it->freq);
        for (int i = 0; i < cur_pattern.size(); i++) {
            s += cur_pattern[i]; 
        }
        s += ":" + freq + "->";
    }

    return s;
}

void lru_state::update_queue(vector<char> pattern) {
    string pattern_key(std::begin(pattern), std::end(pattern));

    auto pos = map[pattern_key];
    auto res = *pos;
    res.freq++;

    queue.erase(pos);
    queue.push_front(res);
    map[pattern_key] = queue.begin();
}

void lru_state::enqueue(vector<char> pattern) {
    string key(std::begin(pattern), std::end(pattern));

    queue.emplace_front(pattern, 1, 1);
    map[key] = queue.begin();

    while (queue.size() > this->capacity) {
        vector<char> rm_pattern = queue.back().pattern;
        string rm_pattern_str(std::begin(rm_pattern), std::end(rm_pattern));
        map.erase(rm_pattern_str);

        queue.pop_back();
    }
}
