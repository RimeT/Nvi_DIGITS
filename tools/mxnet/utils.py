"""
Default Mxnet Ops as helper functions
include get gpu info, calculate loss/accuracy, image format reverse
"""
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
    return mx.test_utils.list_gpus()


def get_num_gpus():
    return len(get_available_gpus())

