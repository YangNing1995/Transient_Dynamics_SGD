from torchvision import datasets, transforms
import torch
import torch.utils.data as data

def get_transform(dataset_name):
    """Define data preprocessing and normalization steps that can be adjusted for different datasets"""
    if dataset_name == 'MNIST':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'CIFAR10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])

def get_dataset(dataset_name, train=True, transform=None, train_num=None, test_num=None):
    """
    Load and return a subset of the dataset based on the dataset name.
    If train_num or test_num is None or exceeds the max samples per class, returns the full dataset.
    """
    if dataset_name == 'MNIST':
        dataset_class = datasets.MNIST
        # MNIST: Train=60000 (approx 6000/class), Test=10000 (1000/class)
        max_per_class = 6000 if train else 1000
    elif dataset_name == 'CIFAR10':
        dataset_class = datasets.CIFAR10
        # CIFAR10: Train=50000 (5000/class), Test=10000 (1000/class)
        max_per_class = 5000 if train else 1000
    else:
        raise ValueError("Unsupported dataset")

    # Load the full dataset
    dataset = dataset_class(root=f'./data', train=train, download=True, transform=transform)

    # Determine the target number of samples per class requested
    target_num = train_num if train else test_num

    # Return full dataset if target_num is None, -1, or >= max samples
    if target_num is None or target_num == -1 or target_num >= max_per_class:
        return dataset

    subset_indices = []
    
    if not isinstance(dataset.targets, torch.Tensor):
        targets = torch.tensor(dataset.targets)
    else:
        targets = dataset.targets

    for i in range(10):  # Applicable to 10-class datasets
        class_indices = (targets == i).nonzero(as_tuple=True)[0]
        subset_indices.extend(class_indices[:target_num])

    return data.Subset(dataset, subset_indices)

def get_data_loaders(dataset_name, train_num, test_num, batch_size):
    """Create and return train and test data loaders, applicable to multiple datasets"""
    transform = get_transform(dataset_name)
    train_dataset = get_dataset(dataset_name, train=True, transform=transform, train_num=train_num, test_num=test_num)
    test_dataset = get_dataset(dataset_name, train=False, transform=transform, train_num=train_num, test_num=test_num)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
 
    return train_loader, test_loader
