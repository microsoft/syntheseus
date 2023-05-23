from syntheseus.interface.bag import Bag


def test_basic_operations() -> None:
    bag_1 = Bag(["a", "b", "a"])
    bag_2 = Bag(["b", "a", "a"])
    bag_3 = Bag(["a", "b"])

    # Test `__contains__`.
    assert "a" in bag_1
    assert "b" in bag_1
    assert "c" not in bag_1

    # Test `__eq__`.
    assert bag_1 == bag_2
    assert bag_1 != bag_3

    # Test `__iter__`.
    assert Bag(bag_1) == bag_1

    # Test `__len__`.
    assert len(bag_1) == 3
    assert len(bag_3) == 2

    # Test `__hash__`.
    assert len(set([bag_1, bag_2, bag_3])) == 2
