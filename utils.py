from matplotlib import pyplot as plt
import torch

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

def show_images(images, titles=None, ncols_per_row=5):
    n = len(images)
    nrows = n//ncols_per_row
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