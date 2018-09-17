from mxnet import gluon
from model_factory import Tower
import utils as digits


class UserModel(Tower):

    def construct_net(self):
        # net_type indicates gluon.nn
        # default net_type is gluon.nn.HybridSequential()
        net = gluon.model_zoo.vision.resnet50_v2()
        return net

    def loss_function(self):
        loss = digits.classification_loss()
        return loss

