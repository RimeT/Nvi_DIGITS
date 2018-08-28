"""
mxent train prototype, apply nn here after User model defined by user
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import gluon, autograd, ndarray
import mxnet as mx
import utils as digits
import mx_data


# logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO)


class Model(object):
    def __init__(self, stage, num_outputs, optimization=None):
        self.stage = stage
        self.user_model = None
        self._initializer = mx.init.Xavier()
        self._optimization = optimization
        self._net = None
        self._trainer = None
        self.num_outputs = num_outputs
        self.dataloader = None
        self.ctx = mx.cpu()
        self.gpu_num = digits.get_num_gpus()
        self.summaries = []
        self.towers = []

    def create_dataloader(self, db_path):
        self.dataloader = mx_data.LoaderFactory.set_source(db_path)
        self.dataloader.num_outputs = self.num_outputs

    def create_model(self, obj_UserModel):
        if digits.get_num_gpus() > 0:
            self.ctx = [mx.gpu(i) for i in digits.get_available_gpus()]
        self.user_model = obj_UserModel(self.num_outputs)
        self._net = self.user_model.construct_net()
        self._net.hybridize()
        self._net.initialize(self._initializer, ctx=self.ctx[-1])  # error occurred when call mx.gpu()
        # trainer
        self._optimization = 'sgd'
        self._trainer = gluon.Trainer(self._net.collect_params(), self._optimization,
                                      {'learning_rate': self.learning_rate()})

    def start_train(self, epoch_num=1):
        loss_func = self.user_model.loss_function()
        data_loader = self.dataloader.get_gluon_loader()
        for epoch in range(epoch_num):
            for batch_num, (data, label) in enumerate(data_loader):
                data = data.as_in_context(self.ctx[-1])
                label = label.as_in_context(self.ctx[-1])
                # ask auto grad to record the forward pass
                with autograd.record():
                    output = self._net(data)
                    loss = loss_func(output, label)
                loss.backward()
                self._trainer.step(data.shape[0])

                # print lss once in a while
                if batch_num % 50 == 0:
                    curr_loss = ndarray.mean(loss).asscalar()
                    print("Epoch: %d; Batch %d; Loss %f" % (epoch, batch_num, curr_loss))

    def summary(self):
        """
        Merge train summaries
        """
        for t in self.towers:
            self.summaries += t.summaries

    def train_batch(self, train_batch, ctx):
        pass

    def valid_batch(self, train_batch, ctx):
        pass

    def add_tower(self, obj_tower, x, y):
        is_training = self.stage == digits.STAGE_TRAIN
        is_inference = self.stage == digits.STAGE_INF
        input_shape = self.dataloader.get_shape()
        tower = obj_tower(x, y, input_shape, self.num_outputs, is_training, is_inference)
        self.towers.append(tower)
        return tower

    def global_step(self):
        pass

    def learning_rate(self):
        return 0.001

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

