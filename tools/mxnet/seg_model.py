"""
mxnet segmentation model.

input:train arguments from tools.mxnet.main

function:
1.Connected to dataloader
2.Model train&validation. Logging in a specific format
3.Save snapshots

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from mxnet import gluon, autograd, ndarray
import mxnet as mx
import utils as digits
from model_factory import ModelFactory
import mx_data


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class SegmentationModel(ModelFactory):

    def __init__(self, lr_base, snaps_dir, snaps_pf, snaps_itv=1, valid_itv=1, optimization=None):
        self.lr = lr_base
        self.user_model = None
        self._initializer = mx.init.Xavier()
        self._optimization = optimization
        self._net = None
        self._trainer = None
        self.train_loader = None
        self.valid_loader = None
        self.log_interval = 100
        self.snapshot_dir = snaps_dir
        self.snapshot_prefix = snaps_pf
        self.snapshot_interval = snaps_itv
        self.valid_interval = valid_itv
        self.ctx = mx.cpu()
        self.gpu_num = digits.get_num_gpus()
        self.summaries = []
        self.towers = []


    def create_dataloader(self, train_db, valid_db):
        logging.info('seg--train=' + train_db)
        logging.info('seg--val='+ valid_db)
        #TODO:create mxnet dataloader, utilize mx_data.LoaderFactory
        #self.train_loader = mx_data.LoaderFactory.set_source(train_db)
        #self.valid_loader = mx_data.LoaderFactory.set_source(valid_db)
        pass


    def create_model(self, obj_UserModel):
        """
        model arguments set here
        input obj_UserModel is a UserModel class defined by user
        """
        if digits.get_num_gpus() > 0:
            self.ctx = [mx.gpu(i) for i in digits.get_available_gpus()]
        self.user_model = obj_UserModel(self.train_loader.num_outputs)
        self._net = self.user_model.construct_net()
        self._net.hybridize()
        self._net.initialize(self._initializer, ctx=self.ctx[0])
        # trainer
        self._optimization = 'sgd'
        self._trainer = gluon.Trainer(self._net.collect_params(), self._optimization,
                                      {'learning_rate': self.learning_rate()})


    def start_train(self, epoch_num=1):
        """
        logging format:

        Started training the model
        Training (epoch 1.0): loss = 0.427018, lr = 0.001000, accuracy = 0.923062
        Validation (epoch 2.0): loss = 0.343974, accuracy = 0.951862
        END

        """
        logging.info("hi from segmentation start_train")
        pass

