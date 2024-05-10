import torch
from torchvision import transforms
from loaders.multi_mnist_loader import MNIST
from loaders.cityscapes_loader import CITYSCAPES
from loaders.segmentation_augmentations import *
from loaders.celeba_loader import CELEBA

from scipy import io
import os.path as osp
from sklearn import preprocessing
import torch.utils.data as data


# Setup Augmentations
cityscapes_augmentations= Compose([RandomRotate(10),
                                   RandomHorizontallyFlip()])

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' in params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'cityscapes' in params['dataset']:
        train_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['train'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']), augmentations=cityscapes_augmentations)
        val_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['val'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'celeba' in params['dataset']:
        train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst
    
    if 'sarcos' in params['dataset']:

        train_dst = SARCOS(root = osp.join(configs['sarcos']['path'],"sarcos_inv.mat"))
        # val_dst = SARCOS(root= osp.join(configs['sarcos']['path'],"sarcos_inv_test.mat"))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        # val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)

        # for revisiting scalarization experiments
        val_dst = train_dst
        val_loader = train_loader
        print('using training as validation set')

        return train_loader, train_dst, val_loader, val_dst

class SARCOS(data.Dataset):
    def __init__(self, root=None):      
        self.X, self.y = self.load_SARCOS_numpy(root)
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        print("SARCOS dataset loaded")
        print("Data shape: ", self.X.shape, self.y.shape)

    def load_SARCOS_numpy(self,file):
        data = io.loadmat(file)[osp.basename(file).split(".")[0]]
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        X = data[:,:21] # 21-dimensional input space (7 joint positions, 7 joint velocities, 7 joint accelerations) 
        y = data[:,[23,24,25]] # predicting the torques of arms 3, 4, 5
        return X, y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index][0], self.y[index][1], self.y[index][2]

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return self.__class__.__name__