batchnorm2d(def passthrough(x, **kwargs):
    return x

def ELUCons(elu, num_channels):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(num_channels)

def pad_down(in_size, out_size, kernel_size, stride):
    """
    Computes padding size for down convolutions.
    """
    p = (((out_size - 1) * stride) - in_size + kernel_size)
    assert p % 2 == 0, "can't fit padding with given parameters: {}, {}, {}, {}".format(in_size, out_size, kernel_size, stride)
    return p // 2

def pad_up(in_size, out_size, kernel_size, stride):
    """
    Computes padding size for up convolutions.
    """
    p = (in_size - 1) * stride + kernel_size - out_size
    assert p % 2 == 0, "can't fit transposed padding with given parameters: {}, {}, {}, {}".format(in_size, out_size, kernel_size, stride)
    return p // 2

class LUConv(nn.Module):
    """
    A single ReLU(batchnorm2d(Conv())) block.
    """
    def __init__(self, num_channels, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=5, padding=2)
        self.bn1 = batchnorm2d(num_channels)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

def N_Convs(num_channels, depth, elu):
    """
    Creates NUM_CHANNELS LUConv blocks.
    """
    layers = []
    for _ in range(depth):
        layers.append(LUConv(num_channels, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, out_channels, elu):
        super(InputTransition, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.bn1 = batchnorm2d(out_channels)
        self.relu1 = ELUCons(elu, out_channels)
    def forward(self, x):
        if x.shape[1] == 1:
            out = x.repeat(1, self.out_channels, 1, 1)
        else:
            # do we want a PRELU here as well?
            out = self.bn1(self.conv1(x))
            # split input in to 16 channels
            out = self.relu1(torch.add(out, x_rep))
        return out

class DownTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, padding, elu, dropout=True):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=padding)
        self.bn1 = batchnorm2d(out_channels)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d(0.05)
        self.ops = N_Convs(out_channels, num_convs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
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

class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, padding, elu, dropout=True):
        super(UpTransition, self).__init__()
        out_channels //= 2 #because of the concat with feature forwarding
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=padding, stride=2)
        self.bn1 = batchnorm2d(out_channels)
        self.do1 = passthrough
        self.do2 = passthrough
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        if dropout:
            self.do1 = nn.Dropout2d(0.05)
        self.ops = N_Convs(out_channels * 2, num_convs, elu)

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
        self.bn1 = batchnorm2d(n)
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
        return self.softmax(out)
