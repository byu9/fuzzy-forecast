'''
Tree Data Structure
'''

from collections import deque


class Binary_Tree_Node:
    __slots__ = (
        '__dict__',
        '_parent',
        '_l_child',
        '_r_child'
    )

    def __init__(self):
        self._parent = None
        self._l_child = None
        self._r_child = None

    @property
    def parent(self):
        return self._parent

    @property
    def ancestors(self):
        '''
        ancestors from the direct parent to the root node
        '''
        p = self._parent

        while p is not None:
            yield p
            p = p.parent

    @property
    def left_child(self):
        return self._l_child

    @property
    def right_child(self):
        return self._r_child



class Binary_Tree:
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

    @property
    def leaves(self):
        yield from self._leaves

    @property
    def root(self):
        return self._root

    def __contains__(self, node):
        return node in self._nodes

    def topological_ordering(self):
        '''
        Perform a level-order traversal (root, left, right ...)
        '''
        if self._root is None:
            return

        queue = deque(maxlen=len(self._nodes))
        queue.append(self._root)

        while queue:
            node = queue.popleft()
            yield node

            if node._l_child is not None:
                queue.append(node._l_child)

            if node._r_child is not None:
                queue.append(node._r_child)


    def add_node(self, node, parent=None, left_side=True):
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
            assert parent in self._nodes, "unrecognized parent"

            if parent in self._leaves:
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
