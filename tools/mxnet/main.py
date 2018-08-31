from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def save_snapshot(save_dir, snapshot_prefix, epoch):
    pass

import argparse
import os
import mxnet as mx
from mxnet import gluon
from model import Model
import utils as digits
import logging


# model
import inspect

train_set_folder = "/vdata/train"
val_set_folder = "/vdata/test"

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

import sys

def main():
    print("mx--tools.main")
    logging.info('tools.mxnet.main is running')
    # arg parse start
    parser = argparse.ArgumentParser(description='Mxnet train params')
    parser.add_argument('--network',
                        type=str,
                        help='network file addr')
    parser.add_argument('--epoch',
                        type=int,
                        help='train epoch')
    parser.add_argument('--networkDirectory',
                        type=str)
    parser.add_argument('--save',
                        type=str)
    parser.add_argument('--snapshotPrefix',
                        type=str)
    parser.add_argument('--snapshotInterval',
                        type=str)
    parser.add_argument('--lr_base_rate',
                        type=str)
    parser.add_argument('--lr_policy',
                        type=str)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--mean',
                        type=str)
    parser.add_argument('--labels_list',
                        type=str)
    parser.add_argument('--train_db',
                        type=str)
    parser.add_argument('--train_labels',
                        type=str)
    parser.add_argument('--validation_db',
                        type=str)
    parser.add_argument('--validation_labels',
                        type=str)
    parser.add_argument('--lr_gamma',
                        type=str)
    parser.add_argument('--lr_stepvalues',
                        type=str)
    parser.add_argument('--lr_power',
                        type=str)
    parser.add_argument('--shuffle',
                        type=int)
    parser.add_argument('--croplen',
                        type=int)
    parser.add_argument('--subtractMean',
                        type=str)
    parser.add_argument('--seed',
                        type=str)
    parser.add_argument('--optimization',
                        type=str)
    parser.add_argument('--validation_interval',
                        type=str)
    parser.add_argument('--log_runtime_stats_per_step',
                        type=str)
    parser.add_argument('--weights',
                        type=str)
    parser.add_argument('--augFlip',
                        type=str)
    parser.add_argument('--augNoise',
                        type=str)
    parser.add_argument('--augContrast',
                        type=str)
    parser.add_argument('--augWhitening',
                        type=str)
    parser.add_argument('--augHSVh',
                        type=str)
    parser.add_argument('--augHSVs',
                        type=str)
    parser.add_argument('--augHSVv',
                        type=str)
    
    args = vars(parser.parse_args())
    # arg parse end
    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                args['networkDirectory'], args['network'])
    exec(open(path_network).read(), globals())

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
    batch_size = int(args['batch_size'])
    train_model = Model(digits.STAGE_TRAIN,
                        num_outputs=num_outputs,
                        optimization='sgd')
    train_model.create_dataloader(db_path=train_set_folder)
    train_model.dataloader.setup(shuffle=True,
                                 batch_size=batch_size,
                                 seed=int(args['seed']))

    #val_model = Model(digits.STAGE_VAL, num_outputs)
    #val_model.create_dataloader(db_path=val_set_folder)
    #val_model.dataloader.setup(shuffle=False,
    #                           batch_size=batch_size,
    #                           seed=42)

    # train
    train_model.create_model(UserModel)

    epoch_num = int(args["epoch"])
    train_model.start_train(epoch_num=epoch_num)
if __name__ ==  '__main__':
    main()

