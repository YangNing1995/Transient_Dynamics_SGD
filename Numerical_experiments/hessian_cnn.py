import torch
import numpy as np
from scipy.io import savemat
from argparse import ArgumentParser
import os
import time
from model import SimpleCNN  
from data_utils import get_data_loaders
from torch.func import functional_call, hessian 

# --- Disable TF32 to prevent numerical overflow (Critical for Hessian) ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def compute_loss_stateless(params, model, data, target, criterion):
    """
    A pure function to compute loss using functional_call.
    """
    # functional_call performs a forward pass using 'params' instead of model.parameters()
    output = functional_call(model, params, (data,))
    return criterion(output, target)

def main(BS_list, LR_list, total_realizations, dataset_name='MNIST', train_num=100, test_num=20, hidden_num=50):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current Dataset: {dataset_name}")

    # --- 1. Handle train_num for Full Dataset (-1 or None) ---
    if train_num is None or train_num == -1:
        if dataset_name == 'MNIST':
            train_num = 6000  # Max per class for MNIST
        elif dataset_name == 'CIFAR10':
            train_num = 5000  # Max per class for CIFAR10
        print(f"Auto-configured train_num to max per class: {train_num}")

    # --- 2. Determine Input Dimension based on Dataset ---
    if dataset_name == 'MNIST':
        input_dim = 784   # 28x28x1
    elif dataset_name == 'CIFAR10':
        input_dim = 3072  # 32x32x3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # --- 3. Prepare Data ---
    # We use a large batch size for Hessian calculation
    hessian_calc_batch_size = 1000
    
    train_loader, _ = get_data_loaders(dataset_name, train_num, test_num, hessian_calc_batch_size)
    
    # Get a single batch for Hessian calculation
    # (Note: For strictly correct full-dataset Hessian, you should use the batch accumulation method 
    # provided in previous answers. Here we use a single large batch as per your template.)
    Train_data, Train_target = next(iter(train_loader))
    Train_data, Train_target = Train_data.to(device), Train_target.to(device)
    print(f"Data shape for Hessian calculation: {Train_data.shape}")

    # --- 4. Initialize Model (SimpleCNN) ---
    model = SimpleCNN(input_dim=input_dim, hidden=hidden_num).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # --- 5. Identify the Specific Target Layer ---
    # User Request: "The weight from the last hidden layer to the output node"
    # Structure of SimpleCNN.classifier:
    # [0] Flatten -> [1] Linear(Feature, Hidden) -> [2] ReLU -> [3] Linear(Hidden, Output)
    # The weight we want is at index [3].
    target_layer_name = 'classifier.3.weight'
    
    # Verify the layer exists
    params = dict(model.named_parameters())
    if target_layer_name not in params:
        # Fallback logic if naming is different
        print(f"Warning: {target_layer_name} not found. Available keys: {list(params.keys())}")
        # Try to find the last weight parameter automatically
        target_layer_name = list(params.keys())[-2] # Usually the last weight (before bias)
    
    target_weight = params[target_layer_name]
    num_params = target_weight.numel() # Get total number of elements (e.g., 50*10 = 500)
    
    print(f"Calculating Hessian for layer: {target_layer_name}")
    print(f"Parameter shape: {target_weight.shape}, Total elements: {num_params}")

    for batch_size in BS_list:
        for learning_rate in LR_list:
            for realization in range(1, total_realizations + 1):
                
                # Setup time points
                max_iteration = int(1000 / learning_rate)
                time_points = np.linspace(0, max_iteration, 101)
                time_points = np.round(time_points).astype(int)
                
                # Initialize storage for eigenvalues using dynamic 'num_params' instead of hardcoded 2500
                H_save = np.zeros([num_params, 101], dtype=np.float32)

                for i, t in enumerate(time_points):
                    # Note: Ensure these checkpoint directories match your training script's naming (e.g. _cnn suffix)
                    # If you saved them in 'save_checkpoint_cnn', change the path below accordingly.
                    load_dir = f'bs{batch_size}_lr{learning_rate}_repeat{realization}'
                    load_path = f'./save_checkpoint_cnn/{load_dir}/iteration_{t}.pt' 
                    
                    # Load the model state
                    try:
                        state_dict = torch.load(load_path)
                        model.load_state_dict(state_dict)
                    except FileNotFoundError:
                        # Fallback to check default dir if _cnn dir fails
                        load_path_alt = f'./save_checkpoint/{load_dir}/iteration_{t}.pt'
                        try:
                            state_dict = torch.load(load_path_alt)
                            model.load_state_dict(state_dict)
                        except FileNotFoundError:
                            print(f"Warning: Checkpoint not found at {load_path}")
                            continue
                    
                    # Prepare the parameter dictionary
                    params = dict(model.named_parameters())
                    
                    # --- Wrapper Function (Closure) ---
                    def loss_wrt_target_layer(w_target):
                        params_copy = params.copy()
                        params_copy[target_layer_name] = w_target
                        return compute_loss_stateless(params_copy, model, Train_data, Train_target, criterion)

                    # --- Compute Hessian Efficiently on GPU ---
                    target_weight = params[target_layer_name]
                    
                    # 1. Compute Hessian
                    H = hessian(loss_wrt_target_layer)(target_weight)
                    H = H.reshape(target_weight.numel(), target_weight.numel())
                    
                    # 2. Move to CPU
                    H_cpu = H.detach().cpu()
                    
                    # 3. Safety Check
                    if torch.isnan(H_cpu).any() or torch.isinf(H_cpu).any():
                        print(f"Warning: Hessian contains NaNs/Infs at BS={batch_size}, T={t}. Skipping.")
                        H_save[:, i] = np.nan
                        continue

                    # 4. Solve Eigenvalues
                    try:
                        H_eig = torch.linalg.eigvalsh(H_cpu).numpy()[::-1]
                        H_save[:, i] = H_eig
                    except RuntimeError as e:
                        print(f"Error computing eigenvalues at T={t}: {e}")
                        H_save[:, i] = np.nan

                print(f'Successfully computed Hessian for BS={batch_size}, LR={learning_rate}, Repeat={realization}')
                
                # Save results
                # Ensure this output directory matches your preference
                save_dir_root = './save_data_cnn'
                save_dir_sub = os.path.join(save_dir_root, f'bs{batch_size}_lr{learning_rate}')
                os.makedirs(save_dir_sub, exist_ok=True)
                
                save_path = os.path.join(save_dir_sub, f'save_hessian_repeat{realization}.mat')
                savemat(save_path, {'Hessian': H_save})

if __name__ == '__main__':
    start_time = time.time()

    # --- Parameter Configuration ---
    BS_list = [500, 200, 100, 50, 20]
    LR_list = [0.002, 0.005, 0.01, 0.02, 0.05]
    total_realizations = 10
    
    # Dataset and Size Configuration
    dataset_name = 'CIFAR10'   # Options: 'MNIST', 'CIFAR10'
    train_num = -1           # -1 for full dataset
    test_num = -1           
    hidden_num = 128          # Ensure this matches what you used in training!
    
    # Run Main
    main(BS_list, LR_list, total_realizations, dataset_name, train_num, test_num, hidden_num)
    
    end_time = time.time()
    print("Runtime: {:.6f} seconds".format(end_time - start_time))