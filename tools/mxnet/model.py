"""
mxent train prototype, apply nn here after User model defined by user
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from mxnet import gluon, autograd, ndarray
import mxnet as mx
import utils as digits
import mx_data


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class Model(object):
    def __init__(self, lr_base, snaps_dir, snaps_pf,optimization=None):
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
        self.ctx = mx.cpu()
        self.gpu_num = digits.get_num_gpus()
        self.summaries = []
        self.towers = []

    def create_dataloader(self, train_db, valid_db):
        self.train_loader = mx_data.LoaderFactory.set_source(train_db)
        self.valid_loader = mx_data.LoaderFactory.set_source(valid_db)

    def create_model(self, obj_UserModel):
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

    def batch_validation(self, val_loader, loss_func, acc_func, volume, week, epoch, epoch_num):
        acc = acc_func
        for batch_num, (data, label) in enumerate(val_loader):
            data = data.as_in_context(self.ctx[0])
            label = label.as_in_context(self.ctx[0])
            output = self._net(data)
            loss = loss_func(output, label)
            curr_loss = ndarray.mean(loss).asscalar()
            predictions = mx.nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)

            if batch_num % self.log_interval == 0:
               self.print_train_stats(2, volume, week, epoch, epoch_num, batch_num, curr_loss, acc.get()[1])
    
    def start_train(self, epoch_num=1):
        #start_time = time.time() # seem to be useless
        loss_func = self.user_model.loss_function()
        t_loader = self.train_loader.get_gluon_loader()
        v_loader = self.valid_loader.get_gluon_loader()
        logging.info('Started training the model')
        volume = self.train_loader.get_volume()
        week = volume / self.train_loader.get_batch_size()
        smoothing_constant = .01

        try:
            for epoch in range(epoch_num):        
                train_acc = mx.metric.Accuracy()
                valid_acc = mx.metric.Accuracy()
                for batch_num, (data, label) in enumerate(t_loader):
                    data = data.as_in_context(self.ctx[0])
                    label = label.as_in_context(self.ctx[0])
                    # ask auto grad to record the forward pass
                    with autograd.record():
                        output = self._net(data)
                        loss = loss_func(output, label)
                    loss.backward()
                    self._trainer.step(data.shape[0])
                    curr_loss = ndarray.mean(loss).asscalar()
                    moving_loss = (curr_loss if ((batch_num == 0) and (epoch == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                    # accuracy
                    preds = mx.nd.argmax(output, axis=1)
                    train_acc.update(preds=preds, labels=label)

                    # print loss once in a while
                    if batch_num % self.log_interval == 0:
                        self.print_train_stats(1, volume, week, epoch, epoch_num, batch_num, moving_loss, train_acc.get()[1])

                #validation
                self.batch_validation(v_loader, loss_func, valid_acc, volume, week, epoch, epoch_num)


                #snapshot save
                self._net.export(self.snapshot_dir + '/' + self.snapshot_prefix, epoch=epoch)

        except (KeyboardInterrupt):
            logging.info('Interrupt signal received.')
        #train_time = time.time() - start_time # seem to be useless
        logging.info('END')


    def print_train_stats(self, log_type, volume, week, epoch, epoch_num, batch_num, loss, accuracy):
        curr_epoch = round(epoch_num * (epoch * week + batch_num) / (epoch_num * week), 2)
        log_str = ''
        if log_type == 1:
            log_str = self.format_train(curr_epoch, loss, accuracy)
        elif log_type == 2:
            log_str = self.format_valid(curr_epoch, loss, accuracy)
        logging.info(log_str)


    def format_train(self, epoch, loss, accuracy):
        # TODO auto convert params to string: loss accuracy learning_rate
        log_str = "Training (epoch " + str(epoch) + "): "
        log_str = log_str + "loss" + " = " + "{:.6f}".format(loss) + ", "
        log_str = log_str + "accuracy" + " = " + "{:.6f}".format(accuracy) + ", "
        log_str = log_str + "lr" + " = " + "{:.6f}".format(self.learning_rate())

        return log_str

    def format_valid(self, epoch, loss, accuracy):
        log_str = "Validation (epoch " + str(epoch) + "): "
        log_str = log_str + "loss" + " = " + "{:.6f}".format(loss) + ", "
        log_str = log_str + "accuracy" + " = " + "{:.6f}".format(accuracy)

        return log_str

    
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

    def global_step(self):
        pass

    def learning_rate(self):
        return self.lr

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

