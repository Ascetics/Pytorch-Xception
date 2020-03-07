import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tsfs

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from xception import Xception, XceptionBackbone, xception


class ClassificationTrainer(object):
    def __init__(self, net, train_data, valid_data, loss_func, optimizer, device):
        super(ClassificationTrainer, self).__init__()
        self.net = net
        self.train_data = train_data
        self.valid_data = valid_data
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        pass

    @staticmethod
    def _get_acc(output, label):
        pred = torch.argmax(output, dim=1)
        acc = (pred == label).type(torch.float).sum().cpu().numpy() / label.shape[0]
        return acc

    def _train(self):
        train_loss = 0
        train_acc = 0
        self.net.train()
        for im, lb in self.train_data:
            im, lb = im.to(self.device), lb.to(self.device)
            output = self.net(im)

            self.optimizer.zero_grad()
            loss = self.loss_func(output, lb)
            train_loss += loss.detach().item()
            loss.backward()
            self.optimizer.step()

            train_acc += self._get_acc(output, lb)
            pass
        train_loss /= len(self.train_data)
        train_acc /= len(self.train_data)
        return train_loss, train_acc

    def _valid(self):
        valid_loss = 0
        valid_acc = 0
        self.net.eval()
        for im, lb in self.valid_data:
            im, lb = im.to(self.device), lb.to(self.device)
            output = self.net(im)

            loss = self.loss_func(output, lb)
            valid_loss += loss.detach().item()

            valid_acc += self._get_acc(output, lb)
            pass
        valid_loss /= len(self.valid_data)
        valid_acc /= len(self.valid_data)
        return valid_loss, valid_acc

    def train(self, epochs=1):
        for e in range(1, epochs + 1):
            epoch_str = '{:s}|Epoch: {:02d}|'.format(str(datetime.now()), e)
            print(epoch_str)

            train_loss, train_acc = self._train()
            train_str = 'Train Loss: {:.4f}|Train Acc: {:.4f}|'.format(train_loss, train_acc)
            print(train_str, end='')

            valid_loss, valid_acc = self._valid()
            valid_str = 'Valid Loss: {:.4f}|Valid Acc: {:.4f}|'.format(valid_loss, valid_acc)
            print(valid_str)
            pass
        pass

    pass


if __name__ == '__main__':
    BATCH_SIZE = 16
    EPOCHS = 10
    torch.autograd.set_detect_anomaly(True)

    tset = FashionMNIST(root='/root/private/torch_datasets', train=True,
                        transform=tsfs.Compose([tsfs.ToTensor()]))
    tdata = DataLoader(tset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    vset = FashionMNIST(root='/root/private/torch_datasets', train=False,
                        transform=tsfs.ToTensor())
    vdata = DataLoader(vset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    dev = torch.device('cuda:5')

    # model = Xception(1, 10)
    model = xception('paper', 1, 10)
    model.to(dev)

    optim = torch.optim.Adam(model.parameters())

    loss_f = nn.CrossEntropyLoss()
    loss_f.to(dev)

    trainer = ClassificationTrainer(model, tdata, vdata, loss_f, optim, dev)
    trainer.train(EPOCHS)
    pass
