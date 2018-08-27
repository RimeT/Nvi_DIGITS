from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def save_snapshot(save_dir, snapshot_prefix, epoch):
    pass


import mxnet as mx
from mxnet import gluon

from model import Model
from mx_lenet import UserModel
import utils as digits

train_set_folder = "/vdata/train"
val_set_folder = "/vdata/test"

# model
import mx_lenet
import inspect

try:
    UserModel
except NameError:
    print('UserModel is not defined')
    exit(-1)
if not inspect.isclass(UserModel):
    print('UserModel is not a class')
    exit(-1)

# data
num_outputs = 10
batch_size = 15
train_model = Model(digits.STAGE_TRAIN,
                    num_outputs=num_outputs,
                    optimization='sgd')
train_model.create_dataloader(db_path=train_set_folder)
train_model.dataloader.setup(shuffle=True,
                             batch_size=batch_size,
                             seed=42)

val_model = Model(digits.STAGE_VAL, num_outputs)
val_model.create_dataloader(db_path=val_set_folder)
val_model.dataloader.setup(shuffle=False,
                           batch_size=batch_size,
                           seed=42)

# train
train_model.create_model(UserModel)

