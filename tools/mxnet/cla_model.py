"""
mxnet train prototype, apply nn here after User model defined by user
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
from mxnet import gluon, autograd, ndarray
import mxnet as mx
import numpy as np
import utils as digits
from model_factory import ModelFactory
import mx_data


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class ClassificationModel(ModelFactory):

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
        #self.gpu_num = digits.get_num_gpus()
        self.summaries = []
        self.towers = []


    def create_dataloader(self, job_type, train_db, valid_db):
        self.train_loader = mx_data.LoaderFactory.set_source(job_type, train_db)
        if valid_db:
            self.valid_loader = mx_data.LoaderFactory.set_source(job_type, valid_db)


    def create_model(self, obj_UserModel):
        if digits.get_num_gpus() > 0:
            self.ctx = [mx.gpu(i) for i in digits.get_available_gpus()]
        self.user_model = obj_UserModel(self.train_loader.num_outputs)
        self._net = self.user_model.construct_net()
        self._net.hybridize()
        self._net.initialize(self._initializer, ctx=self.ctx)
        # trainer
        self._optimization = 'sgd'
        self._trainer = gluon.Trainer(self._net.collect_params(), self._optimization,
                                      {'learning_rate': self.learning_rate()})


    def batch_validation(self, val_loader, loss_func, acc_func, volume, week, epoch, epoch_num):
        acc = acc_func
        average_acc = 0
        average_loss = 0
        
        for batch_num, (data, label) in enumerate(val_loader):
            #data = data.as_in_context(self.ctx[0])
            #label = label.as_in_context(self.ctx[0])
            data_list = gluon.utils.split_and_load(data, self.ctx)
            label_list = gluon.utils.split_and_load(label, self.ctx)
            #output = self._net(data)
            outputs = [self._net(X) for X in data_list]
            losses = [loss_func(output, label) for output, label in zip(outputs, label_list)]
            #loss = loss_func(output, label)
            curr_loss = [ndarray.mean(loss).asscalar() for loss in losses]
            curr_loss = np.mean(curr_loss)
            average_loss += curr_loss
            #predictions = mx.nd.argmax(output, axis=1)
            predictions = [mx.nd.argmax(output, axis=1) for output in outputs]
            for prediction, l in zip(predictions, label_list):
                acc.update(preds=prediction, labels=l)
            average_acc += acc.get()[1]

        average_acc = average_acc / len(val_loader)
        average_loss = average_loss / len(val_loader)
        self.print_train_stats(2, volume, week, epoch, epoch_num, len(val_loader) -1, average_loss, average_acc)
    

    def start_train(self, epoch_num=1):
        loss_func = self.user_model.loss_function()
        t_loader = self.train_loader.get_gluon_loader()
        v_loader = self.valid_loader.get_gluon_loader()
        logging.info('Started training the model')
        t_volume = self.train_loader.get_volume()
        t_week = t_volume / self.train_loader.get_batch_size()
        v_volume = self.valid_loader.get_volume()
        v_week = v_volume / self.valid_loader.get_batch_size()
        smoothing_constant = .01

        try:
            for epoch in range(epoch_num):        
                train_acc = mx.metric.Accuracy()
                valid_acc = mx.metric.Accuracy()
                for batch_num, (data, label) in enumerate(t_loader):
                    #data = data.as_in_context(self.ctx[0])
                    #label = label.as_in_context(self.ctx[0])
                    data_list = gluon.utils.split_and_load(data, self.ctx)
                    label_list = gluon.utils.split_and_load(label, self.ctx)
                    # ask auto grad to record the forward pass
                    with autograd.record():
                        #output = self._net(data)
                        outputs = [self._net(X) for X in data_list]
                        #loss = loss_func(output, label)
                        losses = [loss_func(output, label) for output, label in zip(outputs, label_list)]
                    #loss.backward()
                    for l in losses:
                        l.backward()
                    #self._trainer.step(data.shape[0])
                    self._trainer.step(self.train_loader.get_batch_size())
                    curr_loss = [ndarray.mean(l).asscalar() for l in losses]
                    curr_loss = np.mean(curr_loss)
                    moving_loss = (curr_loss if ((batch_num == 0) and (epoch == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                    # accuracy
                    preds = [mx.nd.argmax(output, axis=1) for output in outputs]
                    for pred, l in zip(preds, label_list):
                        train_acc.update(preds=pred, labels=l)

                    # print loss once in a while - in batch loop
                    if batch_num % self.log_interval == 0:
                        self.print_train_stats(1, t_volume, t_week, epoch, epoch_num, batch_num, moving_loss, train_acc.get()[1])

                #validation - in epoch loop
                if (epoch % self.valid_interval == 0) or (epoch == epoch_num - 1):
                    self.batch_validation(v_loader, loss_func, valid_acc, v_volume, v_week, epoch, epoch_num)

                #snapshot save - in epoch loop
                if (epoch % self.snapshot_interval == 0) or (epoch == epoch_num - 1):
                    self._net.export(os.path.join(self.snapshot_dir, self.snapshot_prefix), epoch=epoch)
                    self.print_snapshot_stats(epoch)

        except (KeyboardInterrupt):
            logging.info('Interrupt signal received.')
        logging.info('END')


    def print_snapshot_stats(self, epoch):
        snapshot_path = str("%s-%04d.params" % (os.path.join(self.snapshot_dir,self.snapshot_prefix), epoch))
        logging.info('Snapshotting to %s', snapshot_path)
        logging.info('Snapshot saved.')

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

    
    def train_batch(self, train_batch, ctx):
        pass


    def valid_batch(self, train_batch, ctx):
        pass


    def learning_rate(self):
        return self.lr


    def optimizer(self):
        if self._optimization == 'sgd':
            pass

