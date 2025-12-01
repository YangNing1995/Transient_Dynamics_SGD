from torchvision import datasets, transforms
import torch.utils.data as data

def get_transform(dataset_name):
    """定义数据预处理和归一化步骤，可以根据不同的数据集进行调整"""
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

def get_dataset(dataset_name, train=True, transform=None, train_num=100, test_num=20):
    """根据数据集名称加载并返回相应的子集"""
    if dataset_name == 'MNIST':
        dataset_class = datasets.MNIST
    elif dataset_name == 'CIFAR10':
        dataset_class = datasets.CIFAR10
    else:
        raise ValueError("Unsupported dataset")

    dataset = dataset_class(root=f'../Numerical_experiments/data', train=train, download=True, transform=transform)
    subset_indices = []

    for i in range(10):  # 适用于10类数据集，如果数据集类别不同，需要调整
        class_indices = (dataset.targets == i).nonzero(as_tuple=True)[0]
        if train:
            subset_indices.extend(class_indices[:train_num])
        else:
            subset_indices.extend(class_indices[:test_num])

    return data.Subset(dataset, subset_indices)

def get_data_loaders(dataset_name, train_num, test_num, batch_size):
    """创建并返回训练和测试数据加载器，适用于多种数据集"""
    transform = get_transform(dataset_name)
    train_dataset = get_dataset(dataset_name, train=True, transform=transform, train_num=train_num, test_num=test_num)
    test_dataset = get_dataset(dataset_name, train=False, transform=transform, train_num=train_num, test_num=test_num)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=test_num * 10, shuffle=False)  # 保证测试集批次大小足够

    return train_loader, test_loader
