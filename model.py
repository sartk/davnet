import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def toy_fwd(n=1):
    x = torch.rand(n, 1, 364, 364)
    model = DAVNet2D()

class DAVNet2D(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.extract_features = VNetDown()
        self.segmentation = VNetUp()
        self.discriminator = DomainClassifier()
        # idea: 2 fully connected with 2048 neurons each
        # or an inception model as a part of the discriminator

    def forward(self, x, lamb, seg_only):
        out16, out32, out64, out128, out256 = self.extract_features(x)
        seg = self.segmentation(out16, out32, out64, out128, out256)
        if seg_only:
            return seg
        features = torch.flatten(out256)
        domain = self.discriminator(GradReversal.apply(features, lamb))
        return seg, domain

def DomainClassifier():
    c = nn.Sequential()
    c.add_module('d_fc1', nn.Linear(256 * 24 * 24, 2048))
    c.add_module('d_bn1', nn.BatchNorm1d(2048))
    c.add_module('d_relu1', nn.ReLU(True))
    c.add_module('d_fc2', nn.Linear(2048, 2048))
    c.add_module('d_bn2', nn.BatchNorm1d(2048))
    c.add_module('d_relu2', nn.ReLU(True))
    c.add_module('d_fc3', nn.Linear(2048, 2))
    c.add_module('d_softmax', nn.LogSoftmax(dim=1))
    return c

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
        x_rep = torch.cat([x] * self.out_channels, 0)
        out = self.relu1(torch.add(out, x_rep))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, elu, padding=0, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(out_channels, num_convs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, elu, padding, dropout=False):
        super(UpTransition, self).__init__()
        out_channels //= 2 #because of the concat with feature forwarding
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=padding, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(out_channels, num_convs, elu)

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
        self.conv1 = nn.Conv2d(in_channels, n, kernel_size=2, padding=2)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, kernel_size=1)
        self.relu1 = ELUCons(elu, n)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to SELF.NUM_CLASSES channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 1).contiguous()
        # flatten
        out = out.view(out.numel() // self.num_classes, self.num_classes)

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
        self.down_tr32 = DownTransition(16, 32, 1, pad_down(364, 182, 2, 2), elu)
        self.down_tr64 = DownTransition(32, 64, 2, pad_down(182, 96, 2, 2), elu)
        self.down_tr128 = DownTransition(64, 128, 3, pad_down(96, 48, 2, 2), elu, dropout=True)
        self.down_tr256 = DownTransition(128, 256, 2, pad_down(48, 24, 2, 2), elu, dropout=True)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        return out16, out32, out64, out128, out256

class VNetUp(nn.Module):
    def __init__(self, elu=True, nll=False):
        super(VNetUp, self).__init__()
        self.up_tr256 = UpTransition(256, 256, 2, pad_up(24, 48, 2, 2), elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, pad_up(48, 96, 2, 2), elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, pad_up(96, 182, 2, 2), elu)
        self.up_tr32 = UpTransition(64, 32, 1, pad_up(182, 364, 2, 2), elu)
        self.out_tr = OutputTransition(32, 2, elu, nll)

    def forward(out16, out32, out64, out128, out256):
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
