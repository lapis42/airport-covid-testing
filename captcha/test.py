import os
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def crop125(image):
    return transforms.functional.crop(image, 0, 0, 50, 140)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Lambda(crop125)
])


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout2d(),
        )
        self.Linear = [nn.Linear(1536, 10) for i in range(6)]

    def forward(self, x):
        out = self.net(x)
        multiout = [l(out) for l in self.Linear]
        return multiout


# check model
data_path = './data'
model = torch.load('model.pth').eval()
fns = os.listdir(data_path)
n_fns = len(fns)
i = 0
data = os.path.join(data_path, fns[i])
img = Image.open(data).convert('RGB')
X = preprocess(img).unsqueeze(0)
y = model(X)
ans = [i.argmax(1).item() for i in y]
print(ans)
plt.imshow(X.squeeze())
plt.show()
