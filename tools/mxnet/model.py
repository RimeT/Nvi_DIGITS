"""
mxent train prototype, apply nn here after User model defined by user
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import gluon
import mxnet as mx
import utils as digits
import mx_data


# logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO)


class Model(object):
    def __init__(self, stage, num_outputs, optimization=None):
        self.stage = stage
        self._initializer = mx.init.Xavier()
        self._optimization = optimization
        self._net = None
        self.num_outputs = num_outputs
        self.dataloader = None
        self.ctx = mx.cpu()
        self.summaries = []
        self.towers = []

    def create_dataloader(self, db_path):
        self.dataloader = mx_data.LoaderFactory.set_source(db_path)
        self.dataloader.num_outputs = self.num_outputs

    def create_model(self, obj_UserModel):
        if digits.get_num_gpus() > 0:
            self.ctx = mx.gpu()
        self._net = obj_UserModel(self.num_outputs).construct_net()
        self._net.hybridize()
        self._net.collect_params().initialize(self._initializer, ctx=self.ctx)

    def add_tower(self, obj_tower, x, y):
        is_training = self.stage == digits.STAGE_TRAIN
        is_inference = self.stage == digits.STAGE_INF
        input_shape = self.dataloader.get_shape()
        tower = obj_tower(x, y, input_shape, self.num_outputs, is_training, is_inference)
        self.towers.append(tower)
        return tower

    def train(self):
        pass

    def summary(self):
        """
        Merge train summaries
        """
        for t in self.towers:
            self.summaries += t.summaries

    def global_step(self):
        pass

    def learning_rate(self):
        pass

    def optimizer(self):
        if self._optimization == 'sgd':
            pass

    def get_tower_lossed(self, tower):
        """
        :param tower: User model with loss function
        :return:  list  of losses
        """


class Tower(object):

    def __init__(self, num_outputs, is_training=True, is_inference=True):
        self.num_outputs = num_outputs
        self.net_type = gluon.nn.HybridSequential()
        self.is_training = is_training
        self.is_inference = is_inference
        self.summaries = []
        self.train = None

    def gradientUpdate(self, grad):
        pass

