#!/usr/bin/env python3

import random
from random import randrange
from pprint import pformat

class LossyStateGraph:

    def __init__(self, capacity, window_size):
        self.graph = [None] * capacity
        self.capacity = capacity
        self.is_stable = False

        self.drift_counter = 0
        self.window_size = window_size

    def get_next_tree_id(self, src):
        r = randrange(self.graph[src].total_weight)
        cur_sum = 0

        for key, val in self.graph[src].neighbors.items():
            cur_sum += val[0]
            if r < cur_sum:
                val[1] = True
                return key

        return -1

    def update(self):
        self.drift_counter += 1
        if self.drift_counter <= self.window_size:
            return

        self.drift_counter = 0

        # lossy count
        for node in self.graph:
            if node is None:
                continue

            for key, val in list(node.neighbors.items()):
                if val[1] > 0:
                    val[0] += val[1]
                    node.total_weight += val[1]

                else:
                    val[0] -= 1
                    node.total_weight -= 1

                    if val[0] == 0:
                        # remove edge
                        del node.neighbors[key]

                # reset the number of hits
                val[1] = 0


    def add_node(self, key):
        self.graph[key] = Node(key)

    def add_edge(self, src, dest):
        if self.graph[src] == None:
            self.add_node(src)
        if self.graph[dest] == None:
            self.add_node(dest)

        src_node = self.graph[src]
        src_node.total_weight += 1

        if dest not in src_node.neighbors.keys():
            src_node.neighbors[dest] = [0, 0]
        src_node.neighbors[dest][0] += 1

    def __str__(self):
        strs = []
        for i in range(0, self.capacity):
            if self.graph[i] == None:
                continue
            strs.append(f"Node {i}, total_weight={self.graph[i].total_weight}")
            strs.append(pformat(self.graph[i].neighbors))

        return '\n'.join(strs)

    def __repr__(self):
        return self.__str__()

class Node:
    def __init__(self, key):
        self.key = key
        self.neighbors = dict() # <tree_id, [weight, num_hit]>
        self.total_weight = 0
