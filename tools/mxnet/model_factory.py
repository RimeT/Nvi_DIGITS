from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from mxnet import gluon

import utils as digits


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class ModelFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_model(train_type, lr_base, snaps_dir, snaps_pf, snaps_itv=1,
                  valid_itv=1, optimization='sgd'):
        if train_type == 'image-classification':
            from cla_model import ClassificationModel
            return ClassificationModel(lr_base, snaps_dir, snaps_pf, snaps_itv, valid_itv, optimization)
        elif train_type == 'image-object-detection':
            from obd_model import ObjectDetectionModel
            return ObjectDetectionModel(lr_base, snaps_dir, snaps_pf, snaps_itv, valid_itv, optimization)
        elif train_type == 'image-segmentation':
            from seg_model import SegmentationModel
            return SegmentationModel(lr_base, snaps_dir, snaps_pf, snaps_itv, valid_itv, optimization)

    def create_dataloader(self, job_type, train_db, valid_db=None):
        raise NotImplementedError

    def create_model(self, obj_UserModel):
        raise NotImplementedError

    def start_train(self, epoch_num):
        raise NotImplementedError

    


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
