import numpy as np
import pytest
from stl.base import RemoveDuplicates
from stl.mesh import Mesh


@pytest.mark.parametrize(
    ('strategy', 'expected'),
    [
        (0, RemoveDuplicates.NONE),
        (1, RemoveDuplicates.SINGLE),
        (2, RemoveDuplicates.ALL),
        (True, RemoveDuplicates.SINGLE),
    ],
)
def test_integer_duplicate_strategy(
    strategy: int, expected: RemoveDuplicates
) -> None:
    data = np.zeros(4, dtype=Mesh.dtype)
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    data['vectors'][0] = triangle
    data['vectors'][1] = triangle
    data['vectors'][2] = triangle + 2
    data['vectors'][3] = triangle + 4
    actual = Mesh(data.copy(), remove_duplicate_polygons=strategy)
    reference = Mesh(data.copy(), remove_duplicate_polygons=expected)
    np.testing.assert_array_equal(actual.vectors, reference.vectors)


def test_unknown_integer_duplicate_strategy_is_disabled() -> None:
    assert RemoveDuplicates.map(99) is RemoveDuplicates.NONE
