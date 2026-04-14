# Evaluating Optimization Algorithms Tailored for SA-PINNs
### Abstract:
In this work we evaluate the effect of various optimization algorithms tailored for physics-informed neural networks (PINNs). The original Self-Adaptive PINN (SA-PINN) framework utilized the Adam optimization algorithm, which is commonly used in traditional deep learning models. As stated in the paper, these off-the-shelf optimization algorithms may not be appropriate for physics-informed networks. We have identified several works that propose alternative optimization algorithms better suited for physics-informed networks, and evaluate their effect on the performance of the SA-PINN framework. We test the Burgers’, Helmholtz, and Allen–Cahn partial differential equations (PDEs), assessing both computational efficiency and accuracy. We evaluate two main alternative optimization approaches against the baseline. First, self-scaled quasi-Newton methods (SSBFGS and SSBroyden), and second, a meta-learned optimizer (Bihlo, 2023), which trains a small neural network to act as the optimizer itself. By testing these optimization algorithms within the SA-PINN framework, we can determine whether better optimization complements or substitutes the self-adaptive weighting mechanism.

---

This repository builds on the implementation of the **Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism** proposed in:
DOI: https://doi.org/10.1016/j.jcp.2022.111722


## Requirements
Code has been updated to use 'python 3.10'. Necessary packages can be found in the `requirements.txt` file.

## Conda Environment Setup (Recommended)

The repository now includes a `requirements.txt` for a modern tested stack.

1. Create and activate a conda environment:

```
conda create -n sa_pinn python=3.10 -y
conda activate sa_pinn
```

2. Install the required packages from the repo root:

```
pip install -r requirements.txt
```

3. (Optional) Verify core package versions:

```
python -c "import tensorflow as tf, numpy as np, scipy; print(tf.__version__, np.__version__, scipy.__version__)"
```

Expected output:

```
2.15.1 1.26.4 1.12.0
```



## TeX Dependencies

**(Debian)** Some plots require TeX packages, you can have them installed using the following command:

```
sudo apt-get -qq install texlive-fonts-recommended texlive-fonts-extra dvipng
```

## Data

The data used in this paper is publicly available in the Raissi implementation of Physics-Informed Neural Networks [found here](https://github.com/maziarraissi/PINNs). It has already been copied into the appropriate directories for utilization in the script files.

## Usage

You can recreate the results of the paper by simply navigating to the desired system (i.e. opening the Burgers folder) and running the .py script in the folder. After opening the Burgers folder, simply run

```
python burgers.py
```

And the training will begin, followed by the plots.

You can change the optimizer using the --optimizer CLI flag. The code currently supports the Adam and Learnable optimizer.

```
python burgers.py --optimizer adam

python burgers.py --optimizer learnable
```


## Citation


```
@article{mcclenny2020self,
  title={Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism},
  author={McClenny, Levi and Braga-Neto, Ulisses},
  journal={arXiv preprint arXiv:2009.04544},
  year={2020}
}

```
