import abc


class BaseNodeTest(abc.ABC):
    """Base class which defines common tests for graph nodes."""

    @abc.abstractmethod
    def get_node(self):
        pass

    def test_node_comparison(self):
        """Test that nodes are equal if and only if they are the same object."""
        node1 = self.get_node()
        node2 = self.get_node()
        assert node1 is not node2
        assert node1 != node2

    @abc.abstractmethod
    def test_nodes_not_frozen(self):
        """Test that the fields of the node can be modified."""
