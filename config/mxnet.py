#tiansong: check mxnet could be import
from __future__ import absolute_import
from . import option_list


def test_mx_import():
    """
    Tests if mxnet can be imported, returns if it went okay and optional error.
    """
    try:
        import mxnet
        return True
    except (ImportError, TypeError):
        return False


mx_enabled = test_mx_import()

if not mx_enabled:
    print('Mxnet support disabled.')

if mx_enabled:
    option_list['mxnet'] = {
        'enabled': True
    }

else:
    option_list['mxnet'] = {
        'enabled': False
    }

