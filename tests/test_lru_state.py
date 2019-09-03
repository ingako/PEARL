import unittest
import sys
sys.path.append('../src')

from LRU_state import *

class TestLruState(unittest.TestCase):
    def test_integration(self):
        lru_state = LRU_state(4, 0)

        s0 = '00000'
        s1 = '00001'
        s2 = '00010'
        s3 = '00011'
        s4 = '00100'
        states = [s0, s1, s2, s0, s3, s4]

        results = []
        for state in states:
            next_state = lru_state.get_closest_state(state)
            if len(next_state) > 0:
                lru_state.enqueue(next_state)
            else:
                lru_state.enqueue(state)
            results.append(str(lru_state))

        actual_result = '\n'.join(results)
        expected_result = """odict_items([('00000', 1)])
odict_items([('00000', 1), ('00001', 1)])
odict_items([('00000', 1), ('00001', 1), ('00010', 1)])
odict_items([('00001', 1), ('00010', 1), ('00000', 2)])
odict_items([('00001', 1), ('00010', 1), ('00000', 2), ('00011', 1)])
odict_items([('00010', 1), ('00000', 2), ('00011', 1), ('00100', 1)])"""

        self.assertEqual(actual_result, expected_result)

if __name__ == '__main__':
    unittest.main()
