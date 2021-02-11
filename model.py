import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torchvision import models
from torchsummary import summary

def toy_fwd(n=1):
    x = torch.rand(n, 1, 344, 344)
    model = DAVNet2D()
    return x, model

class DAVNet2D(nn.Module):

    def __init__(self, classes=4, disc_in=[3, 4, 5, 6]):
        nn.Module.__init__(self)
        self.down = nn.DataParallel(VNetDown())
        self.up = nn.DataParallel(VNetUp(classes))

        C = [16, 32, 64, 128, 256, 256, 128, 64, 32]
        S = [344, 172, 86, 48, 24, 48, 86, 172, 344]
        self.disc = nn.DataParallel(DomainClassifier(num_channels=sum([C[i] for i in disc_in])))
        self.pool = [nn.DataParallel(nn.AvgPool2d(kernel_size=s, stride=1)) if i in disc_in else None
                                    for i, s in enumerate(S)]
        self.disc_in = disc_in

    def forward(self, x, grad_reversal_coef=1, seg_only=False):

        b = x.size(0)
        encoder_x = self.down(x)
        decoder_x = self.up(encoder_x[0], encoder_x[1], encoder_x[2], encoder_x[3], encoder_x[4])
        seg = decoder_x[-1]

        if (not self.training) or seg_only:
            return seg

        F = encoder_x + decoder_x
        x = [self.pool[i](F[i]).view(b, -1) for i in self.disc_in]
        x = torch.cat(x, 1)
        domain = self.disc(GradReversal.apply(x, grad_reversal_coef))

        return seg, domain

    def __str__(self):
        model_stats = summary(self, (1, 344, 344), verbose=0)
        return str(model_stats)

    def feature_MDD(self, source, target):
        S, T = self.down(source), self.down(target)
        return [torch.norm(s.view(s.size(0), -1).mean(0) - t.view(t.size(0), -1).mean(0)).item() for s, t in zip(S, T)]

def sequential(x, funcs):
    for f in funcs:
        x = f(x)
    return x

class DomainClassifier(nn.Module):
    def __init__(self, num_channels=256):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(num_channels, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(2048, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.fc1(x)))
        out = self.relu2(self.bn2(self.fc2(out)))
        out = self.softmax(self.fc3(out).view(x.size(0), 2))
        return out

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, num_channels):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(num_channels)

def pad_down(in_size, out_size, kernel_size, stride):
    p = (((out_size - 1) * stride) - in_size + kernel_size)
    assert p % 2 == 0, "can't fit padding with given parameters: {}, {}, {}, {}".format(in_size, out_size, kernel_size, stride)
    return p // 2

def pad_up(in_size, out_size, kernel_size, stride):
    p = (in_size - 1) * stride + kernel_size - out_size
    assert p % 2 == 0, "can't fit transposed padding with given parameters: {}, {}, {}, {}".format(in_size, out_size, kernel_size, stride)
    return p // 2

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm2d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, num_channels, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(num_channels, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(num_channels, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, out_channels, elu):
        super(InputTransition, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = ELUCons(elu, out_channels)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x_rep = x.repeat(1, self.out_channels, 1, 1)
        out = self.relu1(torch.add(out, x_rep))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, padding, elu, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d(0.1)
        self.ops = _make_nConv(out_channels, num_convs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, padding, elu, dropout=False):
        super(UpTransition, self).__init__()
        out_channels //= 2 #because of the concat with feature forwarding
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=padding, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = passthrough
        self.do2 = passthrough
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d(0.1)
        self.ops = _make_nConv(out_channels * 2, num_convs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_channels, n, elu, nll):
        super(OutputTransition, self).__init__()
        self.num_classes = n
        self.conv1 = nn.Conv2d(in_channels, n, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, kernel_size=1)
        self.relu1 = ELUCons(elu, n)
        if nll:
            self.softmax = nn.LogSoftmax2d()
        else:
            self.softmax = nn.Softmax2d()

    def forward(self, x):
        # convolve 32 down to SELF.NUM_CLASSES channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        #out = out.permute(0, 2, 3, 1).contiguous()
        # flatten
        #out = out.view(out.numel() // self.num_classes, self.num_classes)

        return self.softmax(out)

class GradReversal(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class VNetDown(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNetDown, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 32, 1, pad_down(344, 172, 2, 2), elu)
        self.down_tr64 = DownTransition(32, 64, 2, pad_down(172, 86, 2, 2), elu)
        self.down_tr128 = DownTransition(64, 128, 3, pad_down(86, 48, 2, 2), elu, dropout=True)
        self.down_tr256 = DownTransition(128, 256, 2, pad_down(48, 24, 2, 2), elu, dropout=True)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        return out16, out32, out64, out128, out256

class VNetUp(nn.Module):
    def __init__(self, num_channels=2, elu=True, nll=False):
        super(VNetUp, self).__init__()
        self.up_tr256 = UpTransition(256, 256, 2, pad_up(24, 48, 2, 2), elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, pad_up(48, 86, 2, 2), elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, pad_up(86, 172, 2, 2), elu)
        self.up_tr32 = UpTransition(64, 32, 1, pad_up(172, 344, 2, 2), elu)
        self.out_tr = OutputTransition(32, num_channels, elu, nll)

    def forward(self, in16, in32, in64, in128, in256):
        out256 = self.up_tr256(in256, in128)
        out128 = self.up_tr128(out256, in64)
        out64 = self.up_tr64(out128, in32)
        out32 = self.up_tr32(out64, in16)
        seg = self.out_tr(out32)
        return out256, out128, out64, out32, seg
