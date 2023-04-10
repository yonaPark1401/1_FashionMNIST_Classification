# task 1: fashion mnist (based on torchvision)
import torch
from torchvision import datasets, transforms
import numpy as np

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_set_full = datasets.FashionMNIST('./', train=True, download=True, transform=transforms)
test_set_full = datasets.FashionMNIST('./', train=False, download=True, transform=transforms)

train_perm = np.random.permutation(len(train_set_full))
train_set_full.targets = (train_set_full.targets.unsqueeze(1)[train_perm]).squeeze()
train_set_full.data = train_set_full.data[train_perm]

test_perm = np.random.permutation(len(test_set_full))
test_set_full.targets = (test_set_full.targets.unsqueeze(1)[test_perm]).squeeze()
test_set_full.data = test_set_full.data[test_perm]


# CNN for fashionMNIST
import torch.nn as nn
class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc_2 = nn.Linear(in_features=600, out_features=120)
        self.fc_3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.drop(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

model = FashionMNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = torchmetrics.HingeLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# prepare data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

set_1 = [0, 1, 3, 5, 8]
set_2 = [2, 4, 6, 7, 9]

def get_dataset(idx, dataset):
    idx_mask = (dataset.targets == idx[0]) | (dataset.targets == idx[1]) | (dataset.targets == idx[2]) | (
                dataset.targets == idx[3]) | (dataset.targets == idx[4])
    idx_set = [i for i in range(len(idx_mask)) if idx_mask[i]==True]
    return idx_set


for idx_1, idx_2 in [[set_1, None],
                     [set_2, None],
                     [set_1, set_2]]:

    if not(idx_1==None) and not(idx_2==None):
        train_idx = get_dataset(idx_2, train_set_full)
        test_idx = get_dataset(idx_1, test_set_full)
    else:
        train_idx = get_dataset(idx_1, train_set_full)
        test_idx = get_dataset(idx_1, test_set_full)

    train_loader = DataLoader(train_set_full, batch_size=100, sampler=train_idx)
    test_loader = DataLoader(test_set_full, batch_size=100, sampler=test_idx)

    # train & test
    epochs = 5
    count_iter = 0
    loss_arr = []
    iter_arr = []
    accuracy_arr = []

    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            train = imgs.view(100, 1, 28, 28)

            outputs = model(train)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count_iter += 1

            if not(count_iter % 50):
                with torch.no_grad():
                    correct = 0
                    total = 0

                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)

                        test = imgs.view(100, 1, 28, 28)
                        outputs = model(test)

                        predictions = torch.max(outputs, 1)[1].to(device)  #[0] - val, [1] - inds
                        correct += (predictions==labels).sum()
                        total += len(labels)

                    accuracy = (correct*100 / total).to('cpu')
                    accuracy_arr.append(accuracy)
                    iter_arr.append(count_iter)
                    loss_arr.append(loss.to('cpu'))


    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.plot(iter_arr, loss_arr)
    plt.title('Loss graph')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(iter_arr, accuracy_arr)
    plt.title('Accuracy graph: ')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')

    plt.show()










