#!/usr/bin/env python3

class LossyStateGraph:

    def __init__(self, capacity):
        self.graph = [None] * capacity
        self.capacity = capacity
        self.is_stable = False

    def get_next_tree_id(self):
        pass

    def add_node(self, key):
        self.graph[key] = Node(key, self.capacity)

    def add_edge(self, src, dest):
        if self.graph[src] == None:
            self.add_node(src)
        if self.graph[dest] == None:
            self.add_node(dest)

        self.graph[src].neighbors.append(dest)

    def remove_edge(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

class Node:
    def __init__(self, key, capacity):
        self.key = key
        self.neighbors = []
        self.counts = []
        self.window_counts = []
        self.total_count = 0

if __name__ == '__main__':
    state_graph = LossyStateGraph(5)

    state_graph.add_edge(0, 1)
    state_graph.add_edge(0, 4)
    state_graph.add_edge(1, 2)
    state_graph.add_edge(1, 3)
    state_graph.add_edge(1, 4)
    state_graph.add_edge(2, 3)
    state_graph.add_edge(3, 4)

    # print(str(state_graph))
