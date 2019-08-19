import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import dsutils as ds

'''
The following program is a 1D convolutional neural network
that defines the kernel_size recursively.

Currently, I'm feeding in data that is 2D but I flatten it.
This is problematic and needs to be fixed.

# TODO:
    - port to 2DConv
    - test on MNIST
'''

class Conv1DNet(nn.Module):
        super(Conv1DNet, self).__init__()
        def __init__(self, input_dim=784, output_dim=10, batching_size=16):
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.batching_size = batching_size

        self.delta = input_dim - output_dim
        self.ksizes = ds.auto.shape.get_ksizes(self.delta)

        self.dims = []
        self.x = torch.ones([self.batching_size, 1, self.in_dim])
        self.layers = ds.auto.shape.conv1d_layers(self.x, self.ksizes)

        self.real_layer_count = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.real_layer_count - 3:
                break
            x = F.relu(layer(x))
            # print(x.shape)

        x = x.view(self.batching_size, 1, -1)  # flatten to 1d before pooling
        x = self.model[-3](x)  # pool_layer

        x = x.view(self.batching_size, -1)

        x = self.model[-2](x)  # linear
        x = self.model[-1](x)  # softmax

        return x.double()




class Conv2DNet(nn.Module):
    def __init__(self, input_shape=(28, 28), out_classes=10):
        super(Net, self).__init__()
        self.in_dim = input_shape[0]
        # we want to convolve until n x n > # classes and (n - 1) ** 2 < # classes
        self.out_dim = math.ceil(math.sqrt(out_classes))
        self.delta = self.in_dim - self.out_dim
        self.ksizes = get_ksizes(self.delta)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 24 -> 12
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 8 -> 4
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 2

    dataset = A()
    x, y = dataset.__getitem__(0)

    flat_x = x.flatten()
    print(f'input_dim: {flat_x.shape[0]}')

    batching_size = math.floor(math.log2(len(dataset)))

    validation_split = .3
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    data_loader = DataLoader(dataset, batch_size=batching_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batching_size, sampler=valid_sampler)

    net = Conv1DNet(flat_x.shape[0], y.shape[0], batching_size).double()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    logging_rate = 10
    losses = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for i, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()

            pred = net(x)

            loss = criterion(pred, torch.max(y.long(), 1)[1])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % logging_rate == 0:
                with torch.no_grad():
                    losses.append(loss)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / logging_rate))
                    running_loss = 0.0

    print('Finished Training')
    for loss_val in losses:
        print(f'loss_val {loss_val}\n')

    with torch.no_grad():
        for h, (x, y) in enumerate(test_loader):

            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()


    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    # torch.save(net.state_dict(), './demos/baselines/saved_models/conv_classify.pt')
    # print(losses)
    torch.save(net.state_dict(), './demos/baselines/saved_models/conv_classify.pt')
