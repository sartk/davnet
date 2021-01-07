from model import DAVNet2D
from dataset import *
import matplotlib.pyplot as plt
import numpy as np


model = DAVNet2D()
PATH = sys.argv[1]

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

data = kMRI('valid', balanced=True, group='all')

while True:
    print()
    i = input('Enter Image Index (0 - {})'.format(len(data) - 1))
    image, segmentation, domain = data[i]
    image = image.view(344, 344).numpy()
    seg_pred, dom_pred = model(image.view(1, 1, 344, 344), 0, False)
    seg_pred = seg_pred.view(-1, 344, 344).numpy()
    segmentation = segmentation.view(-1, 344, 344).numpy()
    dom_pred = dom_pred.argmax(-1).view(-1)[0].item()
    domain = domain.view(-1)[0].item()

    print("True Domain: {}, Predicted Domain: {}".format(domain, dom_pred)

    f = plt.figure()
    f.add_subplot(1,3, 1)
    plt.imshow(image, cmap='gray')
    f.add_subplot(1,3, 2)
    plt.imshow(segmentation, cmap='Qualitative')
    f.add_subplot(1,3, 3)
    plt.imshow(seg_pred, cmap='Qualitative')

    plt.show(block=True)
