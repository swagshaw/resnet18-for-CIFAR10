import math
import random
import PIL
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from torchvision.transforms import ToPILImage

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(value=(0.4914, 0.4822, 0.4465)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



if __name__ == "__main__":
    img = Image.open("/home/xiaoyang/Dev/AI6103-assignment/data/cifar-10-batches-py/train/1_10000.jpg")
    img = transform_train(img)
    show = ToPILImage()
    img = show(img)
    plt.imshow(img)
    plt.show()