import mxnet as mx
from mxnet.gluon import nn


class Residual_v2_bottleneck(nn.HybridBlock):
    def __init__(self, channels, same_shape=True):
        super(Residual_v2_bottleneck, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels // 4, 1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(
                channels // 4, 3, padding=1, strides=strides, use_bias=False)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, 1, use_bias=False)
            self.bn4 = nn.BatchNorm()

            if not same_shape:
                self.conv4 = nn.Conv2D(
                    channels, 1, strides=strides, use_bias=False)

    def hybrid_forward(self, F, x):
        out = self.conv1(self.bn1(x))
        out = F.relu(self.bn2(out))
        out = F.relu(self.bn3(self.conv2(out)))
        out = self.bn4(self.conv3(out))
        if not self.same_shape:
            x = self.conv4(x)
        return out + x


class ResNet164_v2(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False):
        super(ResNet164_v2, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(64, 3, 1, 1, use_bias=False))
            # block 2
            for _ in range(27):
                net.add(Residual_v2_bottleneck(64))
            # block 3
            net.add(Residual_v2_bottleneck(128, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(128))
            # block 4
            net.add(Residual_v2_bottleneck(256, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(256))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.Activation('relu'))
            net.add(nn.AvgPool2D(8))
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out