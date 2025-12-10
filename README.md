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
python run_repeat.py --batch_size 10 --learning_rate 0.01 --total_iterations 10000

\\\

If you want pinned versions for reproducibility, tell me which Python and CUDA version you plan to use and I will update \
equirements.txt\ accordingly.
