# pylint: disable=E1101,R,C
import numpy as np
import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import gzip
import pickle
import numpy as np
from torch.autograd import Variable

MNIST_PATH = "s2_mnist.gz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-3


def load_data(path, batch_size):

    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    # TODO normalize dataset
    # mean = train_data.mean()
    # stdv = train_data.std()

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


class VolterraBlock(nn.Module):
    def __init__(self, f0, f1, f2, b_in, b_l):
        super(VolterraBlock, self).__init__()
        
        self.s2_seq = S2Convolution(f0, f1, b_in, b_l, s2_near_identity_grid())
        self.so3_seq = SO3Convolution(f1, f2, b_l, b_l, so3_near_identity_grid())
        
        f = (f1 + f2) // 2
        self.conv3d_0 = nn.Conv3d(f1, f, kernel_size=1)
        self.conv3d_1 = nn.Conv3d(f2, f, kernel_size=1)
        
        self.bn = nn.BatchNorm3d(f, affine=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.s2_seq(x)
        z = torch.mul(y, y)
        z = self.so3_seq(z)
        
        y = self.conv3d_0(y)
        z = self.conv3d_1(z)
        
        z = torch.add(y, z)
        y = self.relu(z)
        
        return y


class S2VolterraNet(nn.Module):

    def __init__(self):
        super(S2VolterraNet, self).__init__()
        
        f0 = 1
        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l = 10
        
        # volterra layer
        self.seq = VolterraBlock(f0, f1, f2, b_in, b_l)
        
        # output layer
        f = (f1 + f2) // 2
        self.out_layer = nn.Linear(f, f_output)

    def forward(self, x):

        x = self.seq(x)
        x = so3_integrate(x)
        x = self.out_layer(x)

        return x


def main():

    train_loader, test_loader, train_dataset, _ = load_data(
        MNIST_PATH, BATCH_SIZE)

    classifier = S2VolterraNet()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
                loss.item()), end="")
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main()
