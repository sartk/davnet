from model import DAVNet2D
from dataset import *
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
import sys
import os
import torch
from utils import DiceLoss, dice_loss_weighted, per_class_dice, dice_loss_fra, py_dice
from scipy.io import loadmat, savemat

def save_mat(var, path):
      savemat(path, var)

PATH = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
dice = DiceLoss(repr='1-')
np = lambda lst: tuple([l.numpy() for l in lst])
cuda = not not sys.argv[2]

if cuda:
    model = DAVNet2D(4, dp=True).cuda()
    checkpoint = torch.load(PATH)
    cpu = lambda x: x.cpu()
else:
    model = DAVNet2D(4, dp=False)
    checkpoint = torch.load(PATH, map_location=lambda storage, location: storage)
    cpu = lambda x: x

pretrained_dict = checkpoint['model_state_dict']

if not cuda:
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)
model.eval()

data = kMRI('valid', balanced=False, group='source')

while True:
    print()
    i = int(input('Enter Image Index (0 - {})'.format(len(data) - 1)))
    image, segmentation, domain, meta = data__getitem__(i, meta=True)
    segmentation = segmentation.view(1, 4, 344, 344)
    if cuda:
        image, segmentation, domain = image.cuda(), segmentation.cuda(), domain.cuda()
    with torch.no_grad():
        seg_pred, dom_pred = model(image.view(1, 1, 344, 344), 0, False)

    save_mat({'input': image.view(1, 344, 344).numpy(), 'prediction': seg_pred.view(4, 344, 344).numpy(), 'target':segmentation.view(4, 344, 344).numpy(), 'meta':meta}, '/data/bigbone6/skamat/francesco.mat')
    print("Weighted dice: {}".format(dice_loss_weighted(seg_pred, segmentation)))
    print("Native per class loss: {}".format(per_class_dice(seg_pred, segmentation, tolist=True)))
    print("Py per class: {}".format(py_dice(seg_pred, segmentation)))
    print("Path:", meta)
    image = cpu(image.view(344, 344)).numpy()
    seg_pred = seg_pred.view(-1, 344, 344)
    seg_confidence, seg_pred = np(torch.max(seg_pred, dim=0))
    segmentation = cpu(segmentation.view(-1, 344, 344).argmax(0)).numpy()

    save_mat({'input': image, 'prediction': seg_pred, 'target':segmentation}, '/data/bigbone6/skamat/francesco2d.mat')

    print("True Domain: {}, Predicted Domain: {}".format(domain, dom_pred))
    dom_pred = dom_pred.argmax(-1).view(-1)[0].item()
    domain = domain.view(-1)[0].item()

    #print("True Domain: {}, Predicted Domain: {}".format(domain, dom_pred))

    f = plt.figure()
    f.add_subplot(1,4, 1)
    plt.imshow(image, cmap='gray')
    f.add_subplot(1,4, 2)
    plt.imshow(segmentation)
    f.add_subplot(1,4, 3)
    plt.imshow(seg_pred)
    f.add_subplot(1,4,4)
    plt.imshow(seg_confidence, cmap='jet')
    plt.colorbar()

    plt.savefig('tmp.png')
    plt.show()
    print('File saved to tmp.png')
