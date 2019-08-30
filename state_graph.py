#!/usr/bin/env python3

from random import randrange
from pprint import pformat

class LossyStateGraph:

    def __init__(self, capacity, window_len):
        self.graph = [None] * capacity
        self.capacity = capacity
        self.is_stable = False

        self.drift_counter = 0
        self.window_len = window_len

    def get_next_tree_id(self, src):
        r = randrange(self.graph[src].total_weight)
        sum = 0

        for key, val in self.graph[src].neighbors.items():
            sum += val[0]
            if r < sum:
                return key

        return -1

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

        self.graph[src].neighbors[dest][0] += 1
        self.graph[src].neighbors[dest][1] = True

    def remove_edge(self):
        pass

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
        self.neighbors = dict() # <tree_id, [weight, is_hit]>
        self.total_weight = 0

if __name__ == '__main__':
    state_graph = LossyStateGraph(5, 5)

    state_graph.add_edge(0, 1)
    state_graph.add_edge(0, 4)
    state_graph.add_edge(1, 2)
    state_graph.add_edge(1, 3)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(2, 3)
    state_graph.add_edge(3, 4)

    print(state_graph.get_next_tree_id(1))
    print(str(state_graph))
