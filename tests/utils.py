import numpy


def to_array(array, round):
    __tracebackhide__ = True

    if not isinstance(array, numpy.ndarray):
        array = numpy.array(array)

    if round:
        array = array.round(round)

    return array


def array_equals(left, right, round=6):
    __tracebackhide__ = True
    left = to_array(left, round)
    right = to_array(right, round)

    message = 'Arrays are unequal:\n%s\n%s' % (left, right)
    if left.size == right.size:
        message += '\nDifference:\n%s' % (left - right)

    assert (left == right).all(), message
