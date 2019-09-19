#!/usr/bin/env python3

import sys, pprint
from collections import OrderedDict

class LRU_state:

    def __init__(self, capacity, distance_threshold):
        self.capacity = capacity
        self.distance_threshold = distance_threshold
        self.state_dict = OrderedDict()

    def enqueue(self, state_list):
        if state_list == None or len(state_list) == 0:
            return

        state = ''.join(state_list)

        if state not in self.state_dict:
            # print(f"Add state: {state}")
            self.state_dict[state] = 0
        self.state_dict[state] += 1
        self.state_dict.move_to_end(state)

        if len(self.state_dict) > self.capacity:
            self.state_dict.popitem(last=False)

    def get_closest_state(self, target_state):
        closest_state = ''
        min_edit_distance = sys.maxsize
        max_freq = 0

        for cur_state, cur_freq in self.state_dict.items():

            cur_edit_distance = 0
            update_flag = True

            for i in range(0, len(target_state)):
                if cur_state[i] == target_state[i]:
                    continue

                # tree with drift must be unset
                if cur_state[i] == '1' and target_state[i] == '2':
                    update_flag = False
                    break

                cur_edit_distance += 1

                if cur_edit_distance > self.distance_threshold \
                        or cur_edit_distance > min_edit_distance:
                    update_flag = False
                    break

            if not update_flag:
                continue

            if min_edit_distance == cur_edit_distance and cur_freq < max_freq:
                continue

            min_edit_distance = cur_edit_distance
            max_freq = cur_freq
            closest_state = cur_state

        return [i for i in closest_state]

    def get_size(self):
        return sys.getsizeof(self.state_dict)

    def __str__(self):
        return pprint.pformat(self.state_dict.items(), indent=4)

    def __repr__(self):
        return self.__str__()
