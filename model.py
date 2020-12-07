import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DAVNet2D(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.extract_features = VNetDown()
        self.segmentation = VNetUp()
        self.discriminate = DomainClassifier()

    def forward(self, x, lamb):
        out16, out32, out64, out128, out256 = self.extract_features(x)
        features = out256.view(-1,)
        seg = self.segmentation(out16, out32, out64, out128, out256)
        domain = self.discriminate(GradReversal.apply(features, lamb))
        return seg, domain

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, num_channels):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(num_channels)

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
        self.bn1 = ContBatchNorm2d(num_channels)

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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, num_convs, elu, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2*in_channels
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm2d(out_channels)
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
    def __init__(self, in_channels, out_channels, num_convs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm2d(out_channels // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = ELUCons(elu, out_channels // 2)
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
    def __init__(self, in_channels, num_classes, elu, nll):
        super(OutputTransition, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, num_classes, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(num_classes)
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, num_classes)
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
        out = out.view(out.numel() // self.num_classes, self/num_classes)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out

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
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        return out16, out32, out64, out128, out256

class VNetUp(nn.Module):
    def __init__(self):
        super(VNetUp, self).__init__()
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, 2, elu, nll)

    def forward(out16, out32, out64, out128, out256):
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
