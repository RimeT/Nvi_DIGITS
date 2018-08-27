# mxnet train file
from __future__ import absolute_import

import os

from .train import TrainTask
import digits
from digits import utils
from digits.utils import subclass, override, constants
import mxnet as mx

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

# Constants
MXNET_MODLE_FILE = 'mxmodel.py'
MXNET_SNAPSHOT_PREFIX = 'snapshot'
TIMELINE_PREFIX = 'timeline'

@subclass
class MxnetTrainTask(TrainTask):
    """
    Trains a mxnet model
    """

    MXNET_LOG = 'mxnet_output.log'

    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """

        # save network description to file
        super(MxnetTrainTask, self).__init__(**kwargs)
        with open(os.path.join(self.job_dir, MXNET_MODEL_FILE), "w") as outfile:            outfile.write(self.network)

        self.pickver_task_mxnet_train = PICKLE_VERSION

        self.current_epoch = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.solver = None

        self.model_file = MXNET_MODEL_FILE
        self.train_file = constants.TRAIN_DB
        self.val_file = constants.VAL_DB
        self.snapshot_prefix = MXNET_SNAPSHOT_PREFIX
        self.log_file = self.MXNET_LOG

    def __getstate__(self):
        state = super(MxnetTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'mxnet_log' in state:
            del state['mxnet_log']

        return state

    def __setstate__(self, state):
        super(MxnetTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None

    # Task overrides
    @override
    def name(self):
        return 'Train Mxnet Model'

    @override
    def before_run(self):
        super(MxnetTrainTask, self).before_run()
        self.mxnet_log = open(self.path(self.MXNET_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        self.displaying_network = False
        self.temp_unrecognized_output = []
        return True

    @override
    def get_snapshot(self, epoch=-1, download=False, frozen_file=False):
        """
        return snapshot file for specified epoch
	@TODO need to modify for mxnet
        """
	return None
