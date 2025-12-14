import torch
import numpy as np
from scipy.io import savemat
from model import FCN
from data_utils import get_data_loaders
from torch.func import functional_call, hessian  # Import functional API
import time
import os

# --- Disable TF32 to prevent numerical overflow (Critical for Hessian) ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# --- Helper Function (Defined Globally) ---
def compute_loss_stateless(params, model, data, target, criterion):
    """
    A pure function to compute loss using functional_call.
    """
    # functional_call performs a forward pass using 'params' instead of model.parameters()
    output = functional_call(model, params, (data,))
    return criterion(output, target)

def main(load_batch_size, load_learning_rate, load_realization, load_iteration_list, total_iterations, 
         dataset_name='MNIST', train_num=100, test_num=20):
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current Dataset: {dataset_name}")

    # --- 1. Handle train_num for Full Dataset (-1 or None) ---
    if train_num is None or train_num == -1:
        if dataset_name == 'MNIST':
            train_num = 6000  # Max per class for MNIST (Total 60,000)
        elif dataset_name == 'CIFAR10':
            train_num = 5000  # Max per class for CIFAR10 (Total 50,000)
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
    
    # Get a single batch for Hessian calculation (Or implement full-batch accumulation if needed)
    Train_data, Train_target = next(iter(train_loader))
    Train_data, Train_target = Train_data.to(device), Train_target.to(device)
    print(f"Data shape for Hessian calculation: {Train_data.shape}")

    # --- 4. Initialize Model ---
    # Initialize Model with correct input_dim
    model = FCN(input_dim=input_dim, hidden=50).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Identify the specific layer name for Hessian calculation (3rd parameter tensor)
    param_names = list(model.state_dict().keys())
    target_layer_name = param_names[2] 
    print(f"Calculating Hessian for layer: {target_layer_name}")

    # Setup save directory
    save_data_dir = os.path.join(f"./save_data_continue_training", f"bs{load_batch_size}_lr{load_learning_rate}_repeat{load_realization}_ct")
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    for load_iteration in load_iteration_list:
        load_checkpoint_dir = os.path.join(f"./save_checkpoint_continue_training", f"bs{load_batch_size}_lr{load_learning_rate}_repeat{load_realization}_ct{load_iteration}")

        time_points = np.linspace(0, total_iterations, 101)
        time_points = np.round(time_points).astype(int)
        
        # Initialize storage for eigenvalues (float32 is sufficient for symmetric matrices)
        H_save = np.zeros([2500, 101], dtype=np.float32)

        for i, t in enumerate(time_points):
            load_path = f'{load_checkpoint_dir}/iteration_{t}.pt'
            
            # Load the model state
            try:
                state_dict = torch.load(load_path)
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                print(f"Warning: Checkpoint not found at {load_path}")
                continue
            
            # Prepare the parameter dictionary for functional_call
            params = dict(model.named_parameters())
            
            # --- Wrapper Function (Closure) ---
            def loss_wrt_target_layer(w_target):
                params_copy = params.copy()
                params_copy[target_layer_name] = w_target
                return compute_loss_stateless(params_copy, model, Train_data, Train_target, criterion)

            # --- Compute Hessian Efficiently ---
            target_weight = params[target_layer_name]
            
            # 1. Compute Hessian on GPU (Fast)
            H = hessian(loss_wrt_target_layer)(target_weight)
            H = H.reshape(target_weight.numel(), target_weight.numel())
            
            # 2. Move to CPU immediately to avoid 'cusolver' errors
            H_cpu = H.detach().cpu()
            
            # 3. Safety Check for NaNs/Infs
            if torch.isnan(H_cpu).any() or torch.isinf(H_cpu).any():
                print(f"Warning: Hessian contains NaNs/Infs at Iter={load_iteration}, T={t}. Skipping.")
                H_save[:, i] = np.nan
                continue

            # 4. Solve Eigenvalues on CPU (Stable)
            try:
                # [::-1] ensures Descending order (Largest to Smallest)
                H_eig = torch.linalg.eigvalsh(H_cpu).numpy()[::-1]
                H_save[:, i] = H_eig
            except RuntimeError as e:
                print(f"Error computing eigenvalues at T={t}: {e}")
                H_save[:, i] = np.nan

        print(f'Successfully computed Hessian for iteration={load_iteration}, BS={load_batch_size}, LR={load_learning_rate}, Repeat{load_realization}')
        
        # Save results
        savemat(f'{save_data_dir}/save_hessian_ct{load_iteration}.mat', {'Hessian': H_save})

if __name__ == '__main__':
    start_time = time.time()

    # --- Parameter Configuration ---
    load_batch_size = 50
    load_learning_rate = 0.05
    load_realization = 1
    total_iterations = 2000
    load_iteration_list = list(range(0, 1001, 20))

    # load_batch_size = 10
    # load_learning_rate = 0.01
    # load_realization = 1
    # total_iterations = 10000
    # load_iteration_list = [0, 500]
    # --- Dataset Configuration ---
    dataset_name = 'MNIST'   # Options: 'MNIST', 'CIFAR10'
    train_num = 100          # Number of samples per class (or -1 for full dataset)
    test_num = 20            # Number of test samples per class

    # Run Main
    main(load_batch_size, load_learning_rate, load_realization, load_iteration_list, total_iterations, 
         dataset_name, train_num, test_num)
    
    end_time = time.time()
    print("Runtime: {:.6f} seconds".format(end_time - start_time))