import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# ---
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print("example data shape: ", example_data.shape)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     fig.tight_layout()
#     plt.imshow(example_data[i][0], cmap = 'gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])

# plt.show()

# ---


class Simple_NN(nn.Module):
    '''
    - input shape : (1, 28, 28)
    - fc1 : apply a linear transformation, output features should be 128
    - fc2 : apply a linear transformation, output features should be 64
    - classifier : apply a linear transformation, output features should be the class size (10)
    - model : input -> fc1 -> relu -> fc2 -> relu -> classifier
    '''

    def __init__(self):
        super(Simple_NN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = F.relu
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)

        return x


model = Simple_NN()
# print(model)


def train(model, scheduler, optim, loss_fn, train_loader, epochs, device):
    for epoch in range(epochs):
        train_loss = 0.0

        # 1) Set the mode
        model.train()

        for batch in tqdm(train_loader):
            # 2) Initialize Gradients
            optim.zero_grad()

            inputs, target = batch

            # 3) Change the device of the input and target to device where model exists
            inputs = inputs.to(device)
            target = target.to(device)

            # 4) Get output
            output = model(inputs)

            # 5) Get loss using loss_fn
            loss = loss_fn(output, target)

            # 6) Do Backpropagation
            loss.backward()  # backprop

            # 7) Update the optimizer (hint : use the argument optim and its method step)
            optim.step()  # batch

            train_loss += loss.detach().item()

        # 8) Update the scheduler
        scheduler.step()  # 1epoch

        train_loss /= len(train_loader.dataset)

        print(f'Epoch : {epoch + 1}, Training Loss: {train_loss}')


device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(f"device is {device}\n")

num_epochs = 10

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

model.to(device)

train(model, scheduler, optimizer, nn.CrossEntropyLoss(),
      train_loader, num_epochs, device)


def test(model, loss_fn, device):
    # Set the mode
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, target = batch

            # change the device of input & target

            inputs = inputs.to(device)
            target = target.to(device)

            # Get output using the model
            output = model(inputs)

            # Get accuracy using the output and target
            # Hint : output shape is [batch_size x 10]
            loss = loss_fn(output, target)

            test_loss += loss.detach().item()

            pred = output.argmax(dim=1)

            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = 100 * correct / len(test_loader.dataset)
            print(f'Test Loss: {test_loss}, \t Accuracy: {accuracy}%')


test(model, nn.CrossEntropyLoss(), device)
