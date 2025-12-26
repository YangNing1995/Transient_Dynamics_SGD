import torch
import numpy as np
from scipy.io import savemat
from model import FCN
from data_utils import get_data_loaders
from torch.func import functional_call, hessian 
import time

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

# --- Modified: Added train_num and test_num as arguments ---
def main(BS_list, LR_list, total_realizations, dataset_name='MNIST', train_num=100, test_num=20):
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
    # We use a large batch size (e.g., 1000) for Hessian calculation to get a stable estimate
    # If train_num is small (e.g., 100 total samples), the loader will just return all of them.
    hessian_calc_batch_size = 1000
    
    # Pass train_num and test_num to get_data_loaders
    train_loader, _ = get_data_loaders(dataset_name, train_num, test_num, hessian_calc_batch_size)
    
    # Get a single batch for Hessian calculation
    Train_data, Train_target = next(iter(train_loader))
    Train_data, Train_target = Train_data.to(device), Train_target.to(device)
    print(f"Data shape for Hessian calculation: {Train_data.shape}")

    # --- 4. Initialize Model ---
    model = FCN(input_dim=input_dim, hidden=50).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Identify the specific layer name for Hessian calculation
    param_names = list(model.state_dict().keys())
    target_layer_name = param_names[2] 
    print(f"Calculating Hessian for layer: {target_layer_name}")

    for batch_size in BS_list:
        for learning_rate in LR_list:
            for realization in range(1, total_realizations + 1):
                
                # Setup time points
                max_iteration = int(100 / learning_rate)
                time_points = np.linspace(0, max_iteration, 101)
                time_points = np.round(time_points).astype(int)
                
                # Initialize storage for eigenvalues
                H_save = np.zeros([2500, 101], dtype=np.float32)

                for i, t in enumerate(time_points):
                    load_dir = f'bs{batch_size}_lr{learning_rate}_repeat{realization}'
                    load_path = f'./save_checkpoint/{load_dir}/iteration_{t}.pt'
                    
                    # Load the model state
                    try:
                        state_dict = torch.load(load_path)
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
                    
                    # 1. Compute Hessian (Heavy computation happens on GPU)
                    H = hessian(loss_wrt_target_layer)(target_weight)
                    H = H.reshape(target_weight.numel(), target_weight.numel())
                    
                    # 2. Move to CPU immediately for safe solving
                    H_cpu = H.detach().cpu()
                    
                    # 3. Safety Check for NaNs/Infs
                    if torch.isnan(H_cpu).any() or torch.isinf(H_cpu).any():
                        print(f"Warning: Hessian contains NaNs/Infs at BS={batch_size}, T={t}. Skipping.")
                        H_save[:, i] = np.nan
                        continue

                    # 4. Solve Eigenvalues on CPU (Stable & Fast enough)
                    try:
                        # [::-1] ensures Descending order (Largest to Smallest)
                        H_eig = torch.linalg.eigvalsh(H_cpu).numpy()[::-1]
                        H_save[:, i] = H_eig
                    except RuntimeError as e:
                        print(f"Error computing eigenvalues at T={t}: {e}")
                        H_save[:, i] = np.nan

                print(f'Successfully computed Hessian for BS={batch_size}, LR={learning_rate}, Repeat={realization}')
                
                # Save results
                save_path = f'./save_data/bs{batch_size}_lr{learning_rate}/save_hessian_repeat{realization}.mat'
                savemat(save_path, {'Hessian': H_save})

if __name__ == '__main__':
    start_time = time.time()

    # --- Parameter Configuration ---
    BS_list = [1000, 500, 200, 100, 50, 20, 10]
    LR_list =  [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    total_realizations = 20
    
    # Dataset and Size Configuration
    dataset_name = 'MNIST'   # Options: 'MNIST', 'CIFAR10'
    train_num = 100          # Number of samples per class (or -1 for full dataset)
    test_num = 20            # Number of test samples per class

    # Run Main
    main(BS_list, LR_list, total_realizations, dataset_name, train_num, test_num)
    
    end_time = time.time()
    print("Runtime: {:.6f} seconds".format(end_time - start_time))