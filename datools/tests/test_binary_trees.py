'''
Unit tests for trees
'''


import unittest
from datools.collections.binary_trees import Binary_Tree, Binary_Tree_Node

class Test_Tree(unittest.TestCase):
        
    def test_basic(self):
        tree = Binary_Tree()
        
        root = Binary_Tree_Node()
        l_child = Binary_Tree_Node()
        r_child = Binary_Tree_Node()
        l_grandchild = Binary_Tree_Node()
        
        tree.add_node(root)
        tree.add_node(l_child, parent=root, left_side=True)
        tree.add_node(r_child, parent=root, left_side=False)
        tree.add_node(l_grandchild, parent=l_child, left_side=True)
        
        self.assertIn(root, tree.nodes)
        self.assertIn(l_child, tree.nodes)
        self.assertIn(r_child, tree.nodes)
        self.assertIn(l_grandchild, tree.nodes)
        
        self.assertNotIn(root, tree.leaves)
        self.assertNotIn(l_child, tree.leaves)
        
        self.assertIn(r_child, tree.leaves)
        self.assertIn(l_grandchild, tree.leaves)
        
        self.assertEqual(tree.root, root)
        
        self.assertEqual(root.parent, None)
        self.assertEqual(root.left_child, l_child)
        self.assertEqual(root.right_child, r_child)
        
        self.assertEqual(l_child.parent, root)
        self.assertEqual(l_child.left_child, l_grandchild)
        self.assertEqual(l_child.right_child, None)
        
        self.assertEqual(r_child.parent, root)
        self.assertEqual(r_child.left_child, None)
        self.assertEqual(r_child.right_child, None)
        
        self.assertEqual(l_grandchild.parent, l_child)
        self.assertEqual(l_grandchild.left_child, None)
        self.assertEqual(l_grandchild.right_child, None)
        
    def test_death_existing_node(self):
        tree = Binary_Tree()
        
        root = Binary_Tree_Node()
        l_child = Binary_Tree_Node()
        r_child = Binary_Tree_Node()
        l_grandchild = Binary_Tree_Node()
        
        tree.add_node(root)
        tree.add_node(l_child, parent=root, left_side=True)
        tree.add_node(r_child, parent=root, left_side=False)
        tree.add_node(l_grandchild, parent=l_child, left_side=True)
        
        with self.assertRaises(AssertionError):
            tree.add_node(l_grandchild)
    
    def test_death_existing_root(self):
        tree = Binary_Tree()
        
        root = Binary_Tree_Node()
        another_root = Binary_Tree_Node()
        
        tree.add_node(root)
        
        with self.assertRaises(AssertionError):
            tree.add_node(another_root, parent=None)
            
    def test_death_existing_l_child(self):
        tree = Binary_Tree()
        
        root = Binary_Tree_Node()
        l_child = Binary_Tree_Node()
        another_l_child = Binary_Tree_Node()
        
        tree.add_node(root)
        tree.add_node(l_child, parent=root, left_side=True)
        
        with self.assertRaises(AssertionError):
            tree.add_node(another_l_child, parent=root, left_side=True)
    
    
    
    
    
    
        
