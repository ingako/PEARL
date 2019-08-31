import unittest

from state_graph import *

class TestStateGraph(unittest.TestCase):
    def test_integration(self):
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

        self.assertEqual(actual_result, expected_result)

if __name__ == '__main__':
    unittest.main()
