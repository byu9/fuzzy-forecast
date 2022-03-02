'''
Tree Data Structure
'''


class Tree_Node:
    __slots__ = (
        '__dict__',
        '_parent',
        '_l_child',
        '_r_child'
    )



class Tree:
    __slots__ = (
        '__dict__',
        '_root',
        '_leaves',
        '_nodes',
    )

    def __init__(self):
        self._root = None
        self._leaves = list()
        self._nodes = list()

    def _add_node(self, node, parent=None, left_side=True):
        '''
        Add new node to tree
        :param node Node: node to add
        :param left_side bool: True if node is on left side of parent
        :param parent Node: parent of node, None for root node
        '''
        assert node not in self._nodes, "node already in tree"

        if self._root is None:
            assert parent is None, "root node shall not have parent"
            self._root = node

        else:
            assert parent is not None, "missing parent"

            self._leaves.remove(parent)
            if left_side:
                assert parent._l_child is None, "existing l_child"
                parent._l_child = node
            else:
                assert parent._r_child is None, "existing r_child"
                parent._r_child = node

        self._nodes.append(node)
        self._leaves.append(node)
        node._l_child = None
        node._r_child = None
        node._parent = parent
