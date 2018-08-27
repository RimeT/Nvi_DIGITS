#tiansong: mxnet framework
from __future__ import absolute_import

import os
import re
import subprocess
import tempfile
import sys

from .errors import NetworkVisualizationError
from .framework import Framework
import digits
from digits import utils
from digits.model.tasks import MxnetTrainTask
from digits.utils import subclass, override, constants

@subclass
class MxnetFramework(Framework):
    NAME = 'Mxnet'
    CLASS = 'mxnet'

    CAN_SHUFFLE_DATA = True
    SUPPORTS_PYTHON_LAYERS_FILE = False
    SUPPORTS_TIMELINE_TRACING = True

    SUPPORTED_SOLVER_TYPES = ['SGD', 'ADADELTA', 'ADAGRAD', 'ADAGRADDA', 'MOMENTUM', 'ADAM', 'FTRL', 'RMSPROP']
    
    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = ['FLIPPING', 'NOISE', 'CONTRAST', 'WHITENING', 'HSV_SHIFTING']

    def __init__(self):
        super(MxnetFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return MxnetTrainTask(framework_id=self.framework_id, **kwargs)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        """
        # digits/standard-network/alexnet.py googlenet.py lenet.py vgg16.py
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.py$' % network, filename)
                if match:
                    with open(path) as infile:
                        return infile.read()
        # return None if not found
        return None

    @override
    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        # return the same string
        return network_desc

    @override
    def get_network_from_previous(self, previous_network, use_same_dataset):
        """
        return new instance of network from previous network
        """
        # note: use_same_dataset is ignored here because for Tensorflow, DIGITS
        # does not change the number of outputs of the last linear layer
        # to match the number of classes in the case of a classification
        # network. In order to write a flexible network description that
        # accounts for the number of classes, the `nClasses` external
        # parameter must be used, see documentation.

        # return the same network description
        return previous_network

    @override
    def get_network_visualization(self, **kwargs):
        # @TODO implement this later
        return None

