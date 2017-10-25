import math

from mxnet import init
from mxnet.gluon import nn


class Bottleneck(nn.HybridBlock):
    def __init__(self, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                interChannels,
                kernel_size=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / interChannels)))
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(
                growthRate,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(
                    math.sqrt(2. / (9 * growthRate))))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.concat(* [x, out], dim=1)
        return out


class SingleLayer(nn.HybridBlock):
    def __init__(self, growthRate):
        super(SingleLayer, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                growthRate,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(
                    math.sqrt(2. / (9 * growthRate))))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.concat(* [x, out], 1)
        return out


class Transition(nn.HybridBlock):
    def __init__(self, nOutChannels):
        super(Transition, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(
                nOutChannels,
                kernel_size=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / nOutChannels)))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.Pooling(out, kernel=(2, 2), stride=(2, 2), pool_type='avg')
        return out


class DenseNet(nn.HybridBlock):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        with self.name_scope():
            self.conv1 = nn.Conv2D(
                nChannels,
                kernel_size=3,
                padding=1,
                use_bias=False,
                weight_initializer=init.Normal(math.sqrt(2. / nChannels)))
            self.dense1 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)

        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        with self.name_scope():
            self.trans1 = Transition(nOutChannels)

        nChannels = nOutChannels
        with self.name_scope():
            self.dense2 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        with self.name_scope():
            self.trans2 = Transition(nOutChannels)

        nChannels = nOutChannels
        with self.name_scope():
            self.dense3 = self._make_dense(growthRate, nDenseBlocks,
                                           bottleneck)
        nChannels += nDenseBlocks * growthRate

        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.fc = nn.Dense(nClasses)

    def _make_dense(self, growthRate, nDenseBlocks, bottleneck):
        layers = nn.HybridSequential()
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.add(Bottleneck(growthRate))
            else:
                layers.add(SingleLayer(growthRate))
        return layers

    def hybrid_forward(self, F, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.Pooling(
            F.relu(self.bn1(out)),
            global_pool=1,
            pool_type='avg',
            kernel=(8, 8))
        out = self.fc(out)
        return out
