import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torchvision import models
from torchsummary import summary
from transition_block import *


class DAVNet2D(nn.Module):
    def __init__(self, classes=4, disc_in=[3, 4, 5, 6], dp=True):
        """
        CLASSES: number of clases
        DISC_IN: which points on the V-Net to feed into domain classifier
        DP: use DataParallel?
        """
        if dp:
            dp = nn.DataParallel
        else:
            dp = lambda x: x
        nn.Module.__init__(self)
        self.down = dp(VNetDown())
        self.up = dp(VNetUp(classes))

        ### IGNORE BELOW WHEN TRYING TO DEBUG SEGMENTATIONS, THIS IS FOR CLASSIFICATION

        C = [16, 32, 64, 128, 256, 256, 128, 64, 32] # num channels at each point
        S = [344, 172, 86, 48, 24, 48, 86, 172, 344] # image sizes at each point

        # Does domain classification only on points specified in DISC_IN
        self.disc = dp(DomainClassifier(num_channels=sum([C[i] for i in disc_in])))
        self.pool = [dp(nn.AvgPool2d(kernel_size=s, stride=1)) if i in disc_in else None
                                    for i, s in enumerate(S)]
        self.disc_in = disc_in

    def forward(self, x, grad_reversal_coef=1, seg_only=False):
        b = x.size(0) # batch size
        encoder_x = self.down(x) # the result of DownVNet
        # passing in all the feature forwards to UpNet
        decoder_x = self.up(encoder_x[0], encoder_x[1], encoder_x[2], encoder_x[3], encoder_x[4])
        seg = decoder_x[-1] # The final segmentation of UpVnet
        if seg_only:
            return seg
        # fed to domain classifier
        F = encoder_x + decoder_x
        x = [self.pool[i](F[i]).view(b, -1) for i in self.disc_in]
        x = torch.cat(x, 1)
        domain = self.disc(GradReversal.apply(x, grad_reversal_coef))
        return seg, domain

    def __str__(self):
        #model_stats = summary(self, (1, 344, 344), verbose=0)
        return ''

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

def toy_fwd(n=1):
    x = torch.rand(n, 1, 344, 344)
    model = DAVNet2D()
    return x, model

class VNetDown(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNetDown, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 32, 2, pad_down(344, 172, 2, 2), elu)
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
        self.up_tr128 = UpTransition(256, 128, 3, pad_up(48, 86, 2, 2), elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 2, pad_up(86, 172, 2, 2), elu)
        self.up_tr32 = UpTransition(64, 32, 2, pad_up(172, 344, 2, 2), elu)
        self.out_tr = OutputTransition(32, num_channels, elu, nll)

    def forward(self, in16, in32, in64, in128, in256):
        out256 = self.up_tr256(in256, in128)
        out128 = self.up_tr128(out256, in64)
        out64 = self.up_tr64(out128, in32)
        out32 = self.up_tr32(out64, in16)
        seg = self.out_tr(out32)
        return out256, out128, out64, out32, seg
