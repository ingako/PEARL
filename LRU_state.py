#!/usr/bin/env python3

from collections import OrderedDict

class LRU_state:

    def __init__(self, capacity):
        self.size = capacity
        self.states = OrderedDict()

    def get(self, key):
        if key not in self.states:
            return -1
        val = self.states[key]
        self.states.move_to_end(key)

        return val

    def put(self, key, val):
        if key in self.states:
            del self.states[key]

        self.states[key] = val
        if len(self.states) > self.size:
            self.states.popitem(last=False)
