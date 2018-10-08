"""
Default Mxnet Ops as helper functions
include get gpu info, calculate loss/accuracy, image format reverse
"""
import os
import mxnet as mx
from mxnet import gluon

STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
STAGE_INF = 'inf'


def classification_loss():
    return gluon.loss.SoftmaxCrossEntropyLoss()


def classicifation_accuracy(prediction, label):
    return None


def get_available_gpus():
    gpus = list()
    for i in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
        gpus.append(int(i))
    #return mx.test_utils.list_gpus()
    return gpus


def get_num_gpus():
    return len(get_available_gpus())

