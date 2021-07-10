import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataSet(Dataset):
    def __init__(self,
                 annotation_file,
                 img_dir,
                 transform=None,
                 target_transform=None):
        self.label = pd.read_csv(os.path.join(img_dir, annotation_file),
                                 header=None).to_numpy()
        n_label = len(self.label)
        self.img_fn = ['{:05}.png'.format(i) for i in range(n_label)]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fn[idx])
        image = Image.open(img_path)
        label = self.label[idx, :]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


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


train_data = MyDataSet('train.csv', 'train', preprocess)
test_data = MyDataSet('test.csv', 'test', preprocess)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

model = MyNetwork()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()


# start train
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = 0
        for i in range(6):
            loss += loss_fn(pred[i], y[:, i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            for i in range(6):
                test_loss += loss_fn(pred[i], y[:, i]).item()
                correct += (pred[i].argmax(1) == y[:, i]).type(
                    torch.float).sum().item()
    test_loss /= num_batches
    correct /= size * 6
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

# save model
torch.save(model, 'model.pth')
