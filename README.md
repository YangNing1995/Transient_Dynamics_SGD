
# Transient Dynamics of SGD

## Brief
Repository for the paper **"Stochastic gradient descent drives escape from sharper valleys during early transient dynamics"**. This codebase contains:
1.  **Numerical Experiments:** PyTorch implementations for training MLP on MNIST and CNN on CIFAR-10, including tools for landscape analysis (Hessian spectrum, continuation training).
2.  **Theoretical Modeling:** MATLAB simulations for the minimal Two-Valley landscape model to verify the noise-driven selection mechanism.

---

## Repository Layout

### 1. Numerical_experiments/ (PyTorch)
Contains training scripts and analysis utilities for deep learning experiments.
* **Core Scripts:**
    * `model.py`: Neural network architecture definitions (FCN for MNIST, SimpleCNN for CIFAR-10).
    * `data_utils.py`: Dataset loaders and transformation helpers.
    * `train.py`: Standard training loop utilities.
* **Experiment Runners:**
    * `run_repeat_mlp.py`: Driver script for repeated MLP training runs on MNIST.
    * `run_repeat_cnn.py`: Driver script for repeated CNN training runs on CIFAR-10.
    * `continue_training.py`: Implements the deterministic continuation protocol (switching to full-batch GD) to probe valley properties.
* **Landscape Analysis:**
    * `hessian_mlp.py`: Computes Hessian eigenspectrum for MLP models.
    * `hessian_cnn.py`: Computes Hessian eigenspectrum for CNN models.
    * `hessian_continue_training.py`: Hessian analysis specifically for continuation trajectories.
* **Data Structure:**
    * `data/`: Stores datasets (e.g., `cifar-10-batches-py`, `MNIST`) and simulation outputs (`save_checkpoint`, `save_data`).

### 2. Two_valleys_model/ (MATLAB)
Contains simulations for the theoretical stochastic differential equation (SDE) model on a 2D landscape.
* **Simulation Scripts:**
    * `P_flat_simulation.m`: Simulates the SGD dynamics to calculate the convergence probability ($P_{flat}$) towards the flatter valley.
    * `Phase_diagram.m`: Generates the phase diagrams for Convergence Probability and Freezing Time under linear noise scaling ($\Sigma \propto H$).
    * `Phase_diagram_C_H_squared.m`: Generates phase diagrams under quadratic noise scaling ($\Sigma \propto H^2$) to test robustness.
* **Data Files:**
    * `*.mat`: Pre-computed simulation results (e.g., `result_PhsDgm_...H2.mat`) used to plot the phase diagrams in the paper.

### 3. Plot_figures/
* Jupyter notebooks for generating the figures in the main text and SI (e.g., PCA projections, loss curves).

---

## Environment Setup

**Recommendation:** Use Python 3.8/3.9 with Conda for GPU support.

### Option A: Conda (Recommended)
```powershell
# Create environment
conda create -n transient python=3.9 -y
conda activate transient

# Install PyTorch with CUDA 11.7 (adjust cuda version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

```

### Option B: Virtualenv + pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate  # On Windows
source .venv/bin/activate # On Linux/Mac
pip install -r requirements.txt

```

**Note on NumPy:** If you encounter errors regarding "NumPy 1.x vs 2.x compatibility", please pin numpy to a 1.x version:

```powershell
pip install "numpy<2.0"

```

---

## Usage Examples

### 1. Numerical Experiments (PyTorch)

Navigate to the experiment folder:

```powershell
cd Numerical_experiments

```

**Run repeated MLP training on MNIST:**

```powershell
python run_repeat_mlp.py --batch_size 10 --learning_rate 0.01 --total_iterations 10000 --train_num 100 --test_num 10

```

**Run repeated CNN training on CIFAR-10:**

```powershell
python run_repeat_cnn.py --dataset_name CIFAR10 --batch_size 50 --learning_rate 0.05

```

**Run Continuation Training (Valley Discovery):**

```powershell
# Resume from a checkpoint using full-batch GD to find the local valley
python continue_training.py --batch_size 1000 --learning_rate 0.05 --total_iterations 2000 --load_batch_size 50 --load_learning_rate 0.05 --load_realization 1 --load_iteration_list "0 20 40 100"

```

### 2. Theoretical Simulations (MATLAB)

Open the `Two_valleys_model/` folder in MATLAB.

* Run `Phase_diagram.m` to reproduce the phase diagrams for the standard noise model.
* Run `P_flat_simulation.m` to run new SDE trajectories and observe valley selection.

---

## Datasets

* **MNIST**: 60,000 train / 10,000 test images.
* **CIFAR-10**: 50,000 train / 10,000 test images.
* Datasets will be automatically downloaded to `Numerical_experiments/data/` if not present.

## Checkpoints & Data Paths

* Scripts are configured to save checkpoints and metrics relative to the `Numerical_experiments/data/` directory by default.
* You can modify `args.save_checkpoint_dir` or `args.save_data_dir` in the arguments to point to custom locations (e.g., external drives).

```

```