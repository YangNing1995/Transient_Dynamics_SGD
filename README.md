# Transient_Dynamics_SGD

## Environment for Numerical_experiments

This project requires Python 3.8+ and the following libraries. It's recommended to use a virtual environment when installing these dependencies.

Core dependencies:

- torch
- torchvision
- numpy
- scipy
- matplotlib
- jupyter
- scikit-learn
- tqdm
- pillow

Quick setup (PowerShell):

\\\powershell
# From the repository root:
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
\\\

PyTorch note:
- For CUDA-enabled installations, visit https://pytorch.org/get-started/locally/ and follow the recommended install command for your CUDA and OS.

Quick import test (run from project root):

\\\powershell
python -c "import os, sys; sys.path.insert(0, os.path.abspath(os.getcwd())); from Numerical_experiments.model import FCN; from Numerical_experiments.data_utils import get_data_loaders; print('Numerical_experiments imports OK')"
\\\

Run example training (from repo root):

\\\powershell
cd ./Numerical_experiments
python run_repeat_MLP.py --batch_size 10 --learning_rate 0.01 --total_iterations 10000 --train_num 100 --test_num 10
python continue_training.py --batch_size 1000 --learning_rate 0.05 --total_iterations 2000 --load_batch_size 50 --load_learning_rate 0.05 --load_realization 1 --load_iteration_list list(range(0, 1001, 20))

python continue_training.py --batch_size 1000 --learning_rate 0.01 --total_iterations 10000 --load_batch_size 10 --load_learning_rate 0.01 --load_realization 1 --load_iteration_list list(range(0, 5001, 500))

\\\

If you want pinned versions for reproducibility, tell me which Python and CUDA version you plan to use and I will update \
equirements.txt\ accordingly.
