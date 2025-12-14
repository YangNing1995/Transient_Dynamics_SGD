import torch
import os
import numpy as np
from scipy.io import savemat
from argparse import ArgumentParser
from model import SimpleCNN 
from data_utils import get_data_loaders
from train import train

def parse_args():
    """Parse command-line arguments"""
    parser = ArgumentParser(description="Image Classifier with Small Sample Training (CNN Version)")
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='Name of dataset')
    parser.add_argument('--total_iterations', type=int, default=1000, help='Total training iterations')
    parser.add_argument('--hidden_num', type=int, default=512, help='Number of units in hidden layers')
    parser.add_argument('--train_num', type=int, default=100, help='Number of samples per class in training set')
    parser.add_argument('--test_num', type=int, default=20, help='Number of samples per class in test set')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--save_checkpoint_dir', type=str, default='./save_checkpoint', help='Directory to save checkpoints')
    parser.add_argument('--save_data_dir', type=str, default='./save_data', help='Directory to save results')
    parser.add_argument('--total_realizations', type=int, default=20, help='Number of realizations for one set of hyperparameters')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Handle dataset size
    if args.train_num is None or args.train_num == -1:
        if args.dataset_name == 'MNIST':
            args.train_num = 6000  
        elif args.dataset_name == 'CIFAR10':
            args.train_num = 5000  
        else:
            print(f"Warning: Unknown dataset {args.dataset_name}, skipping auto-size for train_num.")
        
        print(f"Auto-configured train_num to max per class: {args.train_num}")

    # Setup data loaders
    train_loader, test_loader = get_data_loaders(args.dataset_name, args.train_num, args.test_num, args.batch_size)

    # Determine input dimension for SimpleCNN
    if args.dataset_name == 'MNIST':
        input_dim = 784   # 28 * 28 * 1
    elif args.dataset_name == 'CIFAR10':
        input_dim = 3072  # 32 * 32 * 3
    else:
        input_dim = 784   

    # Change 2: Initialize SimpleCNN instead of FCN
    # Make sure SimpleCNN code handles input_dim logic correctly (as provided in previous answers)
    model = SimpleCNN(input_dim=input_dim, hidden=args.hidden_num).to(device)
    print(f"Model initialized: SimpleCNN with input_dim={input_dim}, hidden={args.hidden_num}")

    criterion = torch.nn.CrossEntropyLoss()
    
    # Ensure root checkpoint directory exists
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    for repeat_idx in range(args.total_realizations):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        # Ensure the sub-directories exist
        # Appended '_cnn' to directory name to distinguish from FCN results if needed, 
        # or you can keep it same if you clean the folder.
        # Here I keep your original naming convention to minimize disruption.
        save_checkpoint_dir = os.path.join(args.save_checkpoint_dir, f"bs{args.batch_size}_lr{args.learning_rate}_repeat{repeat_idx + 1}")
        save_data_dir = os.path.join(args.save_data_dir, f"bs{args.batch_size}_lr{args.learning_rate}")
        os.makedirs(save_checkpoint_dir, exist_ok=True)
        os.makedirs(save_data_dir, exist_ok=True)

        # Change 3: Update Initial Weights Filename
        # If we use 'initial_weight.pt', it might load an old FCN weight into CNN and crash.
        # It is safer to use a distinct name or ensure the directory is clean.
        initial_weight_path = os.path.join(args.save_checkpoint_dir, 'initial_weight_cnn.pt')
        
        if os.path.exists(initial_weight_path):
            print(f"Realization {repeat_idx+1}: Loading initial weights from {initial_weight_path}")
            try:
                model.load_state_dict(torch.load(initial_weight_path))
            except RuntimeError as e:
                print(f"Error loading weights: {e}")
                print("Tip: Check if 'initial_weight_cnn.pt' contains weights from a different model architecture.")
                return 
        else:
            print(f"Realization {repeat_idx+1}: Initial weights not found. Saving current initialization to {initial_weight_path}")
            torch.save(model.state_dict(), initial_weight_path)

        # Training loop setup
        samples_per_epoch = args.train_num * 10
        total_epochs = int(np.ceil(args.total_iterations * args.batch_size / samples_per_epoch))

        # Divide total_iterations into 100 time points
        save_iterations = np.linspace(0, args.total_iterations, 101, dtype=int)

        # Reset metrics for this realization
        metrics = {
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'weight_all': [],  
            'wrong_indices': []  
        }

        # Train
        for epoch in range(0, total_epochs):
            metrics = train(model, device, train_loader, test_loader, criterion, optimizer, epoch, save_checkpoint_dir, save_iterations, metrics)

        metrics['save_iterations'] = save_iterations
        savemat(os.path.join(save_data_dir, f'save_metrics_repeat{repeat_idx + 1}.mat'), metrics)

if __name__ == '__main__':
    main()