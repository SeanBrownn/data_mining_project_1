import unittest
import decision_tree
import pandas as pd

class MyTestCase(unittest.TestCase):
    def test_gini(self):
        table1={
            'class': ['c2', 'c2', 'c2']
        }
        df1=pd.DataFrame(table1)
        self.assertEqual(0, decision_tree.gini(df1))

        table2={
            'class': ['c1', 'c1', 'c2', 'c2', 'c2', 'c2']
        }
        df2=pd.DataFrame(table2)
        self.assertAlmostEqual(0.444, decision_tree.gini(df2), delta=0.001)

    def test_gini_split(self):
        parent={
            'class': ['c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2', 'c2']
        }
        df=pd.DataFrame(parent)

        left_child={
            'class': ['c1', 'c1', 'c1', 'c1', 'c1', 'c2']
        }
        left_subset=pd.DataFrame(left_child)

        right_child={
            'class': ['c1', 'c1', 'c2', 'c2', 'c2', 'c2']
        }
        right_subset=pd.DataFrame(right_child)
        self.assertAlmostEqual(0.361, decision_tree.gini_split(df, left_subset, right_subset),
                               delta=0.001)

    def test_best_split_point(self):
        # only involves 1 attribute but tests ability to find best split point
        data={
            'class': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            'value': [60, 70, 75, 85, 90, 95, 100, 120, 125, 220]
        }
        df=pd.DataFrame(data)
        tree=decision_tree.optimal_node(df, ['value'])
        self.assertEqual(tree, decision_tree.node('value', 97.5))

        data2={
            'x': [1,1,0,1],
            'y': [1,1,0,0],
            'z': [1,0,1,0],
            'class': [1,1,2,2]
        }
        df2=pd.DataFrame(data2)
        tree2=decision_tree.optimal_node(df2, ['x', 'y', 'z'])
        self.assertEqual(tree2, decision_tree.node('y', 0))

if __name__ == '__main__':
    unittest.main()
