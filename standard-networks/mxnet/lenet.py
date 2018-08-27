"""
mxnet lenet model
added by tiansong
"""
from mxnet import gluon
from model import Tower


class UserModel(Tower):

    def symbol(self):
        net = self.net
        with net.name_scope():
            # First convolution
            net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Second convolution
            net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Flatten the output before the fully connected layers
            net.add(gluon.nn.Flatten())
            # First fully connected layers with 512 neurons
            net.add(gluon.nn.Dense(512, activation="relu"))
            # Second fully connected layer with as many neurons as the number of classes
            net.add(gluon.nn.Dense(self.num_outputs))

            return net

    def loss(self):
        loss = None
        return loss

