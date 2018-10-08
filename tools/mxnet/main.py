from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def save_snapshot(save_dir, snapshot_prefix, epoch):
    pass

import argparse
import sys
import inspect
import os
import json
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from model_factory import ModelFactory
from cla_model import ClassificationModel
import utils as digits
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def inference(image_file, json_file, param_file, labels_file, job_dir):
    image = mx.nd.load(image_file)[0]
    transformer = transforms.Compose([
        transforms.Resize(500),
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])
    x = transformer(image).expand_dims(axis=0)
    net = gluon.nn.SymbolBlock.imports(json_file, ['data'], param_file)
    pred = net(x)[0]
    print("shape--"+str(pred.shape))
    confid = mx.ndarray.softmax(pred)
    confid = confid.asnumpy().tolist()
    logging.info('Predictions for image ' + '0' + ": " + json.dumps(confid))


def main():
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
                        type=float)
    parser.add_argument('--lr_base_rate',
                        type=str)
    parser.add_argument('--lr_policy',
                        type=str)
    parser.add_argument('--allPredictions',
                        type=int)
    parser.add_argument('--batch_size',
                        type=int)
    parser.add_argument('--mean',
                        type=str)
    parser.add_argument('--labels_list',
                        type=str)
    parser.add_argument('--train_rec',
                        type=str)
    parser.add_argument('--val_rec',
                        type=str)
    parser.add_argument('--datajob_type',
                        type=str)
    parser.add_argument('--datajob_dir',
                        type=str)
    parser.add_argument('--train_db',
                        type=str)
    parser.add_argument('--train_labels',
                        type=str)
    parser.add_argument('--validation_db',
                        type=str)
    parser.add_argument('--validation_labels',
                        type=str)
    parser.add_argument('--inference_db',
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
    parser.add_argument('--visualize_inf',
                        type=int)
    parser.add_argument('--seed',
                        type=str)
    parser.add_argument('--optimization',
                        type=str)
    parser.add_argument('--validation_interval',
                        type=int)
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
    # test
    logging.info("task_arguments:" + str(args))

    # inference
    if args['inference_db'] is not None:
        image_file = args['inference_db'] # mx.ndarray tempfilepath. file content is a list, list[0] is the image
        json_file = args['network']
        param_file = args['weights']
        labels_file = args['labels_list']
        job_dir = args['save']
        inference(image_file, json_file, param_file, labels_file, job_dir)
        return

    # arg parse end
    path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                args['networkDirectory'], args['network'])
    exec(open(path_network).read(), globals())


    train_db = args['train_db']
    val_db = args['validation_db']
    train_rec = args['train_rec']
    val_rec = args['val_rec']
    lr_base = float(args['lr_base_rate'])
    snaps_dir = args['save']
    snaps_pf = args['snapshotPrefix']
    snaps_itv = args['snapshotInterval']
    valid_itv = args['validation_interval']
    seed = None
    logging.info('train_rec=%s' % train_rec)
    logging.info('val_rec=%s' % val_rec)
    seed = args['seed']

    try:
        UserModel
    except NameError:
        print('UserModel is not defined')
        exit(-1)
    if not inspect.isclass(UserModel):
        print('UserModel is not a class')
        exit(-1)
    
    datajob_type = args['datajob_type']
    logging.info('datajob type is ' + datajob_type)
    datajob_dir = args['datajob_dir']

    batch_size = args['batch_size']
    labels_list = None
    nclasses = None
    if args['labels_list'] is not None:
        labels_list = args['labels_list']
        with open(labels_list) as lf:
            nclasses = len(lf.readlines())
    

    train_model = ModelFactory().get_model(datajob_type,
                                           lr_base,
                                           snaps_dir,
                                           snaps_pf,
                                           snaps_itv,
                                           valid_itv,
                                           optimization='sgd')

    train_model.create_dataloader(datajob_type, train_db=train_rec, valid_db=val_rec)
    train_model.train_loader.setup(shuffle=True,
                                 batch_size=batch_size,
                                 seed=seed,
                                 nclasses=nclasses)
    if train_model.valid_loader:
        train_model.valid_loader.setup(shuffle=False,
                                       batch_size=batch_size,
                                       nclasses=nclasses)

    # train
    train_model.create_model(UserModel)

    epoch_num = int(args["epoch"])
    train_model.start_train(epoch_num=epoch_num)


if __name__ ==  '__main__':
    main()

