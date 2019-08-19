import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import dsutils
from dsutils.auto import run_epoch

'''
This network is defined recursively.
|layers| ~ log_2(input_dim)
'''
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        self.encoder = []
        self.decoder = []

        self.instantiate_network()

        self.enc = nn.ModuleList(self.encoder)
        self.dec = nn.ModuleList(self.decoder)

    def instantiate_network(self):

        prev = self.input_dim
        cur = self.input_dim

        tuples = []

        while cur != 1:
            cur = prev // 2
            tuples.append((prev, cur))
            prev = cur

        print(tuples)

        for tup in tuples:
            self.encoder.append(nn.Linear(tup[0], tup[1]))

        for tup in tuples[::-1]:
            self.decoder.append(nn.Linear(tup[1], tup[0]))


    def forward(self, x, direction):
        # directions: encoding and decoding
        for layer in direction:
            x = F.relu(layer(x))
        return x

    def __repr__(self):
        print(f'encoder: {self.enc}')
        print(f'decoder: {self.dec}')
        return 'network'


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")



    model = VAE(input_dim=).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        run_epoch.train(args, model, device, train_loader, optimizer, epoch)
        run_epoch.test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()


#
# if __name__ == '__main__':
#
#     data_loader = Loader()
#     net = VAE(data_loader.length).double()
#
#     print(net)
#     optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
#     epochs = 1
#
#     VAE.train()
#
#     for i in range(1, epochs + 1):
#         for j, (x, _) in enumerate(data_loader):
#             optimizer.zero_grad()
#
#             encoded = net.forward(x, net.enc)
#             decoded = net.forward(encoded, net.dec)
#
#             loss = torch.abs(x - decoded)
#
#             loss.backward()
#             optimizer.step()
