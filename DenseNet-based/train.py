import torch
import os
import DUnet
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch import nn, optim
import argparse

dataset_path = 'dataset/'

torch.manual_seed(42)

bz = 1


class DiceLoss(nn.Module):
    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
        smooth = 0.01
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class MyDataset(Dataset):
    def __init__(self, x, y, start, end):
        self.x = x[start:end]
        self.y = y[start:end]
        self.data = 0
        self.label = 0

    def __getitem__(self, index):
        self.data = self.x[index]
        self.label = self.y[index]
        self.data = torch.from_numpy(self.data)
        return self.data, self.label

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--dataset', type=str, default='dataset/')
    parser.add_argument('--bz', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=200, type=int)

    args = parser.parse_args()

    print(torch.cuda.is_available())
    device = torch.device('cuda')

    x = np.load(args.dataset + 'training_set.npy')
    y = np.load(args.dataset + 'test_set.npy')

    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x = x[shuffle_index]
    y = y[shuffle_index]

    data_set = MyDataset(x, y, 0, len(x))
    x = None
    y = None
    model = DUnet.DenseNet().to(device)

    train_loader = DataLoader(dataset=data_set, batch_size=args.bz, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dice_loss = DiceLoss().to(device)

    if not os.path.exists('saved_models/'):
        os.mkdir('saved_models/')

    losslist = []
    epochlist = []
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = dice_loss(out, y)
            print(epoch, loss.item())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch_loss / len(data_set))
        state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(state_dict, r"saved_models/" + 'epoch_' + str(epoch) + '.pth')
        losslist.append(epoch_loss / len(data_set))
        epochlist.append(epoch)
