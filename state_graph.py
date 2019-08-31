#!/usr/bin/env python3

import random
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
                val[1] = True
                return key

        return -1

    def update(self):
        self.drift_counter += 1
        if self.drift_counter <= self.window_len:
            return

        self.drift_counter = 0

        # lossy count
        for node in self.graph:
            if node is None:
                continue

            for key, val in list(node.neighbors.items()):
                if val[1]:
                    # is hit
                    val[1] = False

                    val[0] += 1
                    node.total_weight += 1

                else:
                    val[0] -= 1
                    node.total_weight -= 1

                    if val[0] == 0:
                        # remove edge
                        del node.neighbors[key]

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
            src_node.neighbors[dest] = [0, False]
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
        self.neighbors = dict() # <tree_id, [weight, is_hit]>
        self.total_weight = 0

if __name__ == '__main__':
    random.seed(0)
    state_graph = LossyStateGraph(5, 1)
    results = []

    state_graph.add_edge(0, 1)
    state_graph.add_edge(0, 4)
    state_graph.update()

    results.append(str(state_graph) + '\n')

    state_graph.add_edge(1, 2)
    state_graph.add_edge(1, 3)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(1, 4)

    results.append("get next tree id for 1")
    results.append(str(state_graph.get_next_tree_id(1)))

    results.append("Before update")
    results.append(str(state_graph) + '\n')

    state_graph.update()

    results.append("After update")
    results.append(str(state_graph) + '\n')

    state_graph.add_edge(2, 3)
    state_graph.add_edge(3, 4)
    state_graph.update()
    results.append(str(state_graph))

    results.append("get next tree id for 2")
    results.append(str(state_graph.get_next_tree_id(2)))

    results.append("Before update")
    results.append(str(state_graph) + '\n')

    state_graph.update()

    results.append("After update")
    results.append(str(state_graph))

    actual_result = '\n'.join(results)
    expected_result = """Node 0, total_weight=2
{1: [1, False], 4: [1, False]}
Node 1, total_weight=0
{}
Node 4, total_weight=0
{}

get next tree id for 1
4
Before update
Node 0, total_weight=2
{1: [1, False], 4: [1, False]}
Node 1, total_weight=6
{2: [1, False], 3: [1, False], 4: [4, True]}
Node 2, total_weight=0
{}
Node 3, total_weight=0
{}
Node 4, total_weight=0
{}

After update
Node 0, total_weight=0
{}
Node 1, total_weight=5
{4: [5, False]}
Node 2, total_weight=0
{}
Node 3, total_weight=0
{}
Node 4, total_weight=0
{}

Node 0, total_weight=0
{}
Node 1, total_weight=5
{4: [5, False]}
Node 2, total_weight=1
{3: [1, False]}
Node 3, total_weight=1
{4: [1, False]}
Node 4, total_weight=0
{}
get next tree id for 2
3
Before update
Node 0, total_weight=0
{}
Node 1, total_weight=5
{4: [5, False]}
Node 2, total_weight=1
{3: [1, True]}
Node 3, total_weight=1
{4: [1, False]}
Node 4, total_weight=0
{}

After update
Node 0, total_weight=0
{}
Node 1, total_weight=4
{4: [4, False]}
Node 2, total_weight=2
{3: [2, False]}
Node 3, total_weight=0
{}
Node 4, total_weight=0
{}"""

    assert(actual_result == expected_result)
