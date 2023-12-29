# shape: batch size x channels x heights x widths
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm.auto import tqdm

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

CIFAR10_LABEL = ['airplane', 'automobile', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

augment_pool = [
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
]


class CustomDataset(Dataset):
    def __init__(self, train, trnasform=None, data_dir="./data/CIFAR10"):
        self.data = datasets.CIFAR10(root=data_dir, train=train, download=True)
        self.trnasform = trnasform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for x, y in data:
            if self.trnasform is not None:
                x = self.trnasform(x)

            x = transforms.ToTensor()(x)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)

        batch_x = torch.stack(batch_x).float()
        batch_y = torch.cat(batch_y).long()
        return batch_x, batch_y


class RandomAugmentation:
    def __init__(self, augment_pool, prob=0.5):
        self.augment_pool = augment_pool
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:  # prob 확률로 augmentation을 적용
            # augment_pool에서 무작위로 하나의 augmentation 메소드 선택
            augment_method = random.choice(self.augment_pool)
            image = augment_method(image)  # 선택된 augmentation 메소드를 이미지에 적용
        return image


train_dataset = CustomDataset(
    train=True, trnasform=RandomAugmentation(augment_pool, 0.5))
test_dataset = CustomDataset(
    train=False, trnasform=RandomAugmentation(augment_pool, 0.5))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=4
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.reshape(-1, 16 * 6 * 6)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, optimizer, loss_fn=criterion):
    model.train()
    tqdm_bar = tqdm(enumerate(train_loader))
    for batch_idx, (image, label) in tqdm_bar:
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()

            # 예측값과 실제값이 맞는지 확인하여 'correct' 업데이트
            pred = output.max(1, keepdim=True)[1]  # 가장 확률이 높은 클래스를 예측값으로 사용
            # 예측값과 실제값이 일치하는 경우의 수를 더합니다.
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # 평균 손실을 계산하기 위해 데이터셋의 크기로 나눕니다.
    test_accuracy = 100. * correct / len(test_loader.dataset)  # 정확도를 계산합니다.

    return test_loss, test_accuracy


learning_rate = 0.001
num_epochs = 5
batch_size = 128
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=train_dataset.collate_fn
                          )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=test_dataset.collate_fn
                         )

for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \t Model: CNN, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))
