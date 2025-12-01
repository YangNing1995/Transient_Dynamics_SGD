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

def main(BS_list, LR_list, total_realizations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, _ = get_data_loaders('MNIST', 100, 20, 1000)
    Train_data, Train_target = next(iter(train_loader))
    Train_data, Train_target = Train_data.to(device), Train_target.to(device)

    # Initialize the model
    model = FCN(50).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for batch_size in BS_list:
        for learning_rate in LR_list:
            for realization in range(1, total_realizations + 1):
                max_iteration = int(100/learning_rate)
                time_points = np.linspace(0, max_iteration, 101)
                time_points = np.round(time_points).astype(int)
                H_save = np.zeros([2500, 101], dtype = 'complex_')

                for i in range(len(time_points)):
                    t = time_points[i]
                    load_dir = f'bs{batch_size}_lr{learning_rate}_repeat{realization}'
                    load_path = f'../save_checkpoint_v2/{load_dir}/iteration_{t}.pt'
                    model.load_state_dict(torch.load(load_path))
                    Train_output = model(Train_data)
                    Train_loss = criterion(Train_output, Train_target)
                    weight_list = list(model.parameters())[2]

                    # Hessian
                    H = hessian(Train_loss, weight_list)  # compute Hessian  
                    H_eig = torch.linalg.eig(H)[0].cpu().numpy()
                    H_save[:, i] = H_eig  

                print(f'Sucessfully compute Hessian for BS={batch_size}, LR={learning_rate}, Repeat{realization}')
                savemat(f'../save_data_v2/bs{batch_size}_lr{learning_rate}/save_hessian_repeat{realization}.mat', {'Hessian': H_save})

if __name__ == '__main__':
    start_time = time.time()
    # 对每个条件最后时刻计算Hessian
    #BS_list = [1000, 500, 200, 100, 50, 20, 10]
    #BS_list = [500, 200, 100, 50, 20, 10]
    BS_list = [10]
    LR_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    total_realizations = 20

    main(BS_list, LR_list, total_realizations)
    end_time = time.time()
    print("运行时间: {:.6f} 秒".format(end_time - start_time))