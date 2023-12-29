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

# ex1)
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=1, shuffle=False)

# # 5th
# image_index = 4
# image, label = train_dataset[image_index]

# plt.imshow(image.permute(1, 2, 0))
# plt.show()

# label_name = CIFAR10_LABEL[label]
# print("The label of this image is:", label_name)

augment_transform = transforms.Compose([
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
])

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

# ex2)
# train_dataset = CustomDataset(train=True)
# test_dataset = CustomDataset(train=False)

# transform = transforms.ToTensor()

# fig = plt.figure()
# for i in range(4):
#     print(f"<Image no.{i}>")
#     image, label_idx = train_dataset[i]
#     image = transform(image)  # 이미지를 텐서로 변환
#     print(image.shape)  # 이미지의 크기 출력
#     label = CIFAR10_LABEL[label_idx]  # 레이블 정보 출력
#     print(label)

# batch_size = 8
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           collate_fn=train_dataset.collate_fn
#                           )
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=batch_size,
#                          shuffle=False,
#                          collate_fn=test_dataset.collate_fn
#                          )


def visualize_batch(batch, augment=None):
    images, labels = batch
    batch_size = images.shape[0]
    pltsize = 2
    plt.figure(figsize=(batch_size * pltsize, pltsize))

    def on_key(event):
        if event.key == 'q':
            plt.close()

    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.axis('off')
        plt.imshow(np.transpose(
            augment(images[i]) if augment else images[i], (1, 2, 0)))
        plt.title('Class: ' + str(CIFAR10_LABEL[labels[i].item()]))
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# ex 2)
# sample_batch = next(iter(train_loader))
# visualize_batch(sample_batch, augment=transforms.Grayscale(
#     num_output_channels=3))


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


BATCH_SIZE = 8
train_dataset = CustomDataset(
    train=True, trnasform=RandomAugmentation(augment_pool, 0.5))
test_dataset = CustomDataset(
    train=False, trnasform=RandomAugmentation(augment_pool, 0.5))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=train_dataset.collate_fn
                          )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         collate_fn=test_dataset.collate_fn
                         )

sample_batch = next(iter(train_loader))
visualize_batch(sample_batch)

# def train(model, train_loader, optimizer, scheduler=None):
#     model.train()
#     train_loss = 0
#     correct = 0
#     tqdm_bar = tqdm(train_loader)
#     for batch_idx, (image, label) in enumerate(tqdm_bar):
#         image = image.to(DEVICE)
#         label = label.to(DEVICE)
#         optimizer.zero_grad()
#         output = model(image)
#         loss = criterion(output, label)
#         loss.backward()
#         train_loss += loss.item()
#         prediction = output.max(1, keepdim=True)[1]
#         correct += prediction.eq(label.veiw_as(prediction)).sum().item()
#         optimizer.step()
#         tqdm_bar.set_description(
#             "Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))
#     if scheduler is not None:
#         scheduler.step()
#     train_loss /= len(train_loader.dataset)
#     train_acc = 100. * correct / len(train_loader.dataset)
#     return train_loss, train_acc
