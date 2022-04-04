from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np



class MNISTDataset(Dataset):

    def __init__(self):

        self.X = np.load("/home/konrad/fun/data/mnist_imgs.npy") / 255.
        self.y = np.load("/home/konrad/fun/data/mnist_labels.npy", allow_pickle = True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return transforms.Normalize(mean = .5, std = .5)(torch.Tensor(self.X[idx].reshape(1, 28, 28))), torch.tensor(int(self.y[idx]))
