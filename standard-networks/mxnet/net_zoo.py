from mxnet import gluon
from model_factory import Tower
import utils as digits
import mxnet.gluon.model_zoo.vision as zoo


class UserModel(Tower):

    def construct_net(self):
        # net_type indicates gluon.nn
        net = zoo.resnet50_v1(classes=self.num_outputs)
        # zoo.resnet50_v2()
        # zoo.alexnet()
        # zoo.densenet121()
        # zoo.mobilenet_v2_1_0()
        # zoo.vgg16()
        # zoo.vgg19_bn()
        return net

    def loss_function(self):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        # loss = gluon.loss.L1Loss()
        # loss = gluon.loss.L2Loss()
        # loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        # loss = gluon.loss.LogisticLoss()
        # loss = gluon.loss.HingeLoss()
        # loss = gluon.loss.HuberLoss()
        # loss = gluon.loss.SoftmaxCELoss()
        return loss

