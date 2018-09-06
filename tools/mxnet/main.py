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

    train_db = args['train_db']
    val_db = args['validation_db']
    lr_base = float(args['lr_base_rate'])
    snaps_dir = args['save']
    snaps_pf = args['snapshotPrefix']
    seed = None
    if 'seed' in args:
        seed = args['seed']

    try:
        UserModel
    except NameError:
        print('UserModel is not defined')
        exit(-1)
    if not inspect.isclass(UserModel):
        print('UserModel is not a class')
        exit(-1)

    # data
    batch_size = None
    if 'batch_size' in args:
        batch_size = args['batch_size']
    train_model = Model(lr_base,
                        snaps_dir,
                        snaps_pf,
                        optimization='sgd')
    train_model.create_dataloader(train_db=train_db, valid_db=val_db)
    train_model.train_loader.setup(shuffle=True,
                                 batch_size=batch_size,
                                 seed=seed)

    train_model.valid_loader.setup(shuffle=False,
                                   batch_size=batch_size)

    # train
    train_model.create_model(UserModel)

    epoch_num = int(args["epoch"])
    train_model.start_train(epoch_num=epoch_num)
if __name__ ==  '__main__':
    main()

