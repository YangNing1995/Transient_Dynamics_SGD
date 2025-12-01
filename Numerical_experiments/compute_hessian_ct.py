import torch
import numpy as np
from scipy.io import savemat
from model import FCN
from data_utils import get_data_loaders
import time

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.flatten())
        #jac.append(grad_x.reshape(x.shape)) 
        #print(y.shape + x.shape)
        grad_y[i] = 0.                                                                                
    return torch.stack(jac)#.reshape(y.shape + x.shape)      

def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)  

def main(load_batch_size, load_learning_rate, load_realization, load_iteration_list, total_iterations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, _ = get_data_loaders('MNIST', 100, 20, 1000)
    Train_data, Train_target = next(iter(train_loader))
    Train_data, Train_target = Train_data.to(device), Train_target.to(device)

    # Initialize the model
    model = FCN(50).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    save_data_dir = os.path.join(f"../save_data_continue_training", f"bs{load_batch_size}_lr{load_learning_rate}_repeat{load_realization}_ct")

    for load_iteration in load_iteration_list:
        load_checkpoint_dir = os.path.join(f"../save_checkpoint_continue_training", f"bs{load_batch_size}_lr{load_learning_rate}_repeat{load_realization}_ct{load_iteration}")

        time_points = np.linspace(0, total_iterations, 101)
        time_points = np.round(time_points).astype(int)
        H_save = np.zeros([2500, 101], dtype = 'complex_')

        for i in range(len(time_points)):
            t = time_points[i]
            load_path = f'{load_checkpoint_dir}/iteration_{t}.pt'
            model.load_state_dict(torch.load(load_path))
            Train_output = model(Train_data)
            Train_loss = criterion(Train_output, Train_target)
            weight_list = list(model.parameters())[2]

            # Hessian
            H = hessian(Train_loss, weight_list)  # compute Hessian  
            H_eig = torch.linalg.eig(H)[0].cpu().numpy()
            H_save[:, i] = H_eig  

        print(f'Sucessfully compute Hessian for iteration={load_iteration}, LR={learning_rate}, Repeat{realization}')
        savemat(f'{save_data_dir}/save_metrics_ct{load_iteration}.mat', {'Hessian': H_save})

if __name__ == '__main__':
    start_time = time.time()

    load_batch_size = 50
    load_learning_rate = 0.05
    load_realization = 14
    total_iterations = 2000
    load_iteration_list = list(range(0, 1001, 20))

    main(load_batch_size, load_learning_rate, load_realization, load_iteration_list, total_iterations)
    end_time = time.time()
    print("运行时间: {:.6f} 秒".format(end_time - start_time))