import torch
from torchvision import datasets, transforms


def data_generator(root, batch_size):
    train_set = datasets.CIFAR10(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   # transforms.Resize([16, 16]),
                                   transforms.Scale([24, 24]),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) # for cifar10
                                   # transforms.Normalize((0.1307,), (0.3081,))  # for mnists
                               ]))
    test_set = datasets.CIFAR10(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  # transforms.Resize([16, 16]),
                                  transforms.Scale([24, 24]),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
                                  # transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
