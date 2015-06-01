from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys


IS_PYTHON2 = (sys.version_info[0] == 2)


def b(s, encoding='ascii', errors='replace'):  # pragma: no cover
    if IS_PYTHON2:
        return bytes(s)
    else:
        return bytes(s, encoding, errors)
