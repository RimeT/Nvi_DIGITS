# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_framework import CaffeFramework
from .framework import Framework
from .torch_framework import TorchFramework
from digits.config import config_value

__all__ = [
    'Framework',
    'CaffeFramework',
    'TorchFramework',
]

if config_value('tensorflow')['enabled']:
    from .tensorflow_framework import TensorflowFramework
    __all__.append('TensorflowFramework')

# added by tiansong
if config_value('mxnet')['enabled']:
    from .mxnet_framework import MxnetFramework
    __all__.append('MxnetFramework')

#
#  create framework instances
#

# torch is optional
torch = TorchFramework() if config_value('torch')['enabled'] else None

# tensorflow is optional
tensorflow = TensorflowFramework() if config_value('tensorflow')['enabled'] else None

# added by tiansong
# mxent is optional
mxnet = MxnetFramework() if config_value('mxnet')['enabled'] else None

# caffe is mandatory
caffe = CaffeFramework()

#
#  utility functions
#


def get_frameworks():
    """
    return list of all available framework instances
    there may be more than one instance per framework class
    """
    #frameworks = [caffe]
    #if mxnet:
    #    frameworks.append(mxnet)
    #if torch:
    #    frameworks.append(torch)
    #if tensorflow:
    #    frameworks.append(tensorflow)
    
    # modify by tiansong
    frameworks = []
    if mxnet:
        frameworks.append(mxnet)
    if tensorflow:
        frameworks.append(tensorflow)
    return frameworks


def get_framework_by_id(framework_id):
    """
    return framework instance associated with given id
    """
    for fw in get_frameworks():
        if fw.get_id() == framework_id:
            return fw
    return None
