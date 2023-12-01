import torch.nn as nn
import torch
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):

    """Basic Block for resnet 18 and resnet 34

    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, mastermodel, in_channels, out_channels, stride=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),  # org LeakyRelu
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.mastermodel = mastermodel

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], num_classes=6, mixup_hidden=False, shake_shake=False, avg_output=False, output_dim=256, preprocessstride=1, resfirststride=1, inchan=3):
        super().__init__()
        img_chan = inchan
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_chan, 64, kernel_size=3, padding=1,
                      bias=False, stride=preprocessstride),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.in_channels = 64
        self.mixup_hidden = mixup_hidden
        self.layer1 = self._make_layer(
            block, 64, num_block[0], resfirststride)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)
        self.conv6_x = nn.Identity() if output_dim <= 0 else self.conv_layer(
            512, output_dim, 1, 0)
        self.conv6_is_identity = output_dim <= 0
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if output_dim > -1:
            self.output_dim = output_dim
        else:
            self.output_dim = 512 * block.expansion

        self.avg_output = avg_output
        self.shake_shake = shake_shake
        self.inplanes = 64
        self.num_classes = num_classes
        widen_factor = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64*widen_factor, num_block[0])
        # self.layer2 = self._make_layer(
        #     block, 128*widen_factor, num_block[1], stride=2)
        # self.layer3 = self._make_layer(
        #     block, 256*widen_factor, num_block[2], stride=2)
        # self.layer4 = self._make_layer(
        #     block, 512*widen_factor, num_block[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * widen_factor, num_classes)

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        print("conv layer input", input_channel, "output", output_channel)
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,
                      kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        print("Making resnet layer with channel", out_channels,
              "block", num_blocks, "stride", stride)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(None, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lam=0.1, target=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the taerget class.

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * \
                lam.expand_as(target_reweighted) + target_shuffled_onehot * \
                (1 - lam.expand_as(target_reweighted))
            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            if torch.max(inp) >= num_classes:
                raise ValueError(
                    "Value in input tensor is greater than or equal to num_classes")

            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to("cuda:0")

        if self.mixup_hidden:
            layer_mix = np.random.randint(0, 3)
        else:
            layer_mix = 0

        out = x

        if lam is not None and target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if lam is not None and target is not None and self.mixup_hidden and layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if lam is not None and target is not None and self.mixup_hidden and layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.layer2(out)
        if lam is not None and target is not None and self.mixup_hidden and layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.layer3(out)
        out = self.layer4(out)

        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # if self.avg_output:
        #     output = self.avg_pool(out)
        #     output = output.view(out.size(0), -1)
        if lam is None or target is None:
            return out
        else:
            return out, target_reweighted
