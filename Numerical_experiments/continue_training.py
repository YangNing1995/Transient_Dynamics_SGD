import torch
import os
import numpy as np
from scipy.io import savemat
from argparse import ArgumentParser
from model import FCN
from data_utils import get_data_loaders
from train import train

def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser(description="Image Classifier with Small Sample Training")
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='Name of dataset')
    parser.add_argument('--total_iterations', type=int, default=2000, help='Total training iterations')
    parser.add_argument('--hidden_num', type=int, default=50, help='Number of units in hidden layers')
    parser.add_argument('--train_num', type=int, default=100, help='Number of samples per class in training set')
    parser.add_argument('--test_num', type=int, default=20, help='Number of samples per class in test set')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')

    parser.add_argument('--load_checkpoint_dir', type=str, default='../save_checkpoint_v2', help='Directory to load checkpoints')
    parser.add_argument('--load_batch_size', type=int, default=50, help='Batch size to load for continue training')
    parser.add_argument('--load_learning_rate', type=int, default=0.05, help='Learning rate to load for continue training')
    parser.add_argument('--load_realization', type=int, default=14, help='Realization index to load for continue training')
    parser.add_argument('--load_iteration_list', nargs='+', type=int, default=list(range(0, 1001, 20)), help='Iterations to load for continue training (e.g., "0 20 1000")')
    parser.add_argument('--save_checkpoint_dir', type=str, default='../save_checkpoint_continue_training', help='Directory to save checkpoints')
    parser.add_argument('--save_data_dir', type=str, default='../save_data_continue_training', help='Directory to save results')

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data loaders
    train_loader, test_loader = get_data_loaders(args.dataset_name, args.train_num, args.test_num, args.batch_size)

    # Initialize the model
    model = FCN(args.hidden_num).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for load_iteration in args.load_iteration_list:
        # Ensure the checkpoint and data directory exists
        save_checkpoint_dir = os.path.join(args.save_checkpoint_dir, f"bs{args.load_batch_size}_lr{args.load_learning_rate}_repeat{args.load_realization}_ct{load_iteration}")
        save_data_dir = os.path.join(args.save_data_dir, f"bs{args.load_batch_size}_lr{args.load_learning_rate}_repeat{args.load_realization}_ct")
        os.makedirs(save_checkpoint_dir, exist_ok=True)
        os.makedirs(save_data_dir, exist_ok=True)

        # Load checkpoint for each iteration in iteration list
        load_checkpoint_path = os.path.join(args.load_checkpoint_dir, f"bs{args.load_batch_size}_lr{args.load_learning_rate}_repeat{args.load_realization}" , f'iteration_{load_iteration}.pt')
        model.load_state_dict(torch.load(load_checkpoint_path))

        # Training loop
        total_epochs = int(args.total_iterations * args.batch_size / (args.train_num * 10))

        # 保存前100个Iteration及total_iterations均匀分为100时间点
        save_iterations = np.linspace(0, args.total_iterations, 101, dtype=int)

        # 保存指标
        metrics = {
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'weight_all': [],  
            'wrong_indices': []  
        }

        for epoch in range(0, total_epochs):
            metrics = train(model, device, train_loader, test_loader, criterion, optimizer, epoch, save_checkpoint_dir, save_iterations, metrics)

        metrics['save_iterations'] = save_iterations
        savemat(os.path.join(save_data_dir, f'save_metrics_ct{load_iteration}.mat'), metrics)

if __name__ == '__main__':
    main()
