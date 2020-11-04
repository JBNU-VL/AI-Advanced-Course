import warnings
from matplotlib import pyplot as plt

import torch
from torch import nn

from torchvision import transforms as T
from torchvision import datasets as D



class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def invest_size(inputs, layer):
    print(f'{"input shape":^70}')
    print(f'{str(inputs.shape):^70}')
    print(f'{"|":^70}')
    print(f'{"V":^70}')
    print(f'{"layer":^70}')
    print(f'{str(layer):^70}')
    print(f'{"|":^70}')
    print(f'{"V":^70}')
    print(f'{"output shape":^70}')
    print(f'{str(layer(inputs).shape):^70}')
    
def sample_random_data(dataset, num=10):
    indices = torch.randint(0, len(dataset), size=(num,))
    data = [dataset[idx] for idx in indices]
    data = list(map(list, zip(*data)))
    return torch.stack(data[0]), data[1]

def fetch_data(dataset, indices):
    data = [dataset[idx] for idx in indices]
    data = list(map(list, zip(*data)))
    return torch.stack(data[0]), data[1]

def show_images(images, titles=None, ncols_per_row=5):
    n = len(images)
    nrows = n//ncols_per_row
    if nrows < 1:
        nrows = 1
    figs, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_per_row,
        figsize=(15, 4*nrows)
    )
    if titles is None:
        titles = [''] * n
    for axis, image, title in zip(axes.ravel(), images, titles):
        axis.axis('off')
        axis.set_title(str(title), fontdict={'fontsize': 15})
        axis.imshow(image.squeeze(), cmap='gray')
    plt.show()

    
@torch.enable_grad()
def train_step(model, input, target, optimizer, criterion, device=None):
    if device is None:
        device = input.device
    model.train()
    optimizer.zero_grad()
    input, target = input.to(device), target.to(device)
    out = model(input)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test_step(model, input, target=None, criterion=None, device=None):
    if device is None:
        device = input.device
    
    model.eval()
    input = input.to(device)
    if target is not None:
        target = target.to(device)
    out = model(input)
    
    loss = None
    if criterion is not None:
        loss = criterion(out, target)
    
    return out, loss

def get_cifar10_dataset(random_crop=False):
    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    if random_crop:
        train_transform = T.Compose([
            T.RandomResizedCrop(32, scale=[0.5, 1.0]),
            T.ToTensor()
        ])
    else:
        train_transform = transform
    
    datasets = {
        'train': D.CIFAR10(
            'data', train=True, transform=train_transform, download=True
        ),
        'test': D.CIFAR10(
            'data', train=False, transform=transform
        )
    }
    return datasets

def make_dataloader(datasets: dict, batch_size: int):
    loader = {
        key: torch.utils.data.DataLoader(
            datasets[key],
            batch_size=batch_size,
            shuffle=key=='train'
        )
        for key in datasets
    }
    return loader
    

def simulate_scheduler(gamma, num_epochs, schd=torch.optim.lr_scheduler.ExponentialLR, **kwargs):
    lrs = []
    dummy_model = nn.Linear(1, 1)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), 0.1)
    dummy_scheduler = schd(
        dummy_optimizer, gamma=gamma, **kwargs
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(num_epochs):
            dummy_scheduler.step()
            lrs.append(dummy_scheduler.get_lr()[0])
    
    return lrs