# Evaluating Optimization Algorithms Tailored for SA-PINNs
### Abstract:
In this work we evaluate the effect of various optimization algorithms tailored for physics-informed neural networks (PINNs). The original Self-Adaptive PINN (SA-PINN) framework utilized the Adam optimization algorithm, which is commonly used in traditional deep learning models. As stated in the paper, these off-the-shelf optimization algorithms may not be appropriate for physics-informed networks. We have identified several works that propose alternative optimization algorithms better suited for physics-informed networks, and evaluate their effect on the performance of the SA-PINN framework. We test the Burgers’, Helmholtz, and Allen–Cahn partial differential equations (PDEs), assessing both computational efficiency and accuracy. We evaluate two main alternative optimization approaches against the baseline. First, self-scaled quasi-Newton methods (SSBFGS and SSBroyden), and second, a meta-learned optimizer (Bihlo, 2023), which trains a small neural network to act as the optimizer itself. By testing these optimization algorithms within the SA-PINN framework, we can determine whether better optimization complements or substitutes the self-adaptive weighting mechanism.

---

This repository builds on the implementation of the **Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism** proposed in:
DOI: https://doi.org/10.1016/j.jcp.2022.111722
GitHub: https://github.com/levimcclenny/SA-PINNs


## Requirements
Code has been updated to use 'python 3.10'. Necessary packages can be found in the `requirements.txt` file.

Replace the _minimize.py and _optimize.py files in the .../site-packages/scipy folder in your conda environment with the modified versions found in the `Optimizers` folder. 

## Conda Environment Setup (Recommended)

The repository now includes a `requirements.txt` for an updated tested stack.

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

Each PDE is trained from its own directory. From the repo root, for example:

```
cd Burgers && python burgers.py
```

Training runs in two phases: **phase 1** updates the PINN weights (and self-adaptive masks where applicable) for `--tf-iter` epochs; **phase 2** refines **only the network weights** with L-BFGS or, when selected, SciPy quasi-Newton, for up to `--newton-iter` iterations. Results (CSVs and figures) are written under `Results/<PDE name>/`.

### Training scripts

| Script | Directory |
|--------|-----------|
| Burgers | `Burgers/burgers.py` |
| Allen–Cahn | `Allen-Cahn/AC.py` |
| Helmholtz | `Helmholtz/helmholtz.py` |
| Helmholtz (NTK variant) | `Helmholtz/helmholtz-NTK.py` |

### Command-line flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--optimizer` | choice | `adam` | Phase-1 optimizer for **network** weights (see below). |
| `--tf-iter` | int | `100` | Number of phase-1 epochs (Adam or learnable updates). |
| `--newton-iter` | int | `100` | Maximum phase-2 iterations (L-BFGS or quasi-Newton on network weights only). |

Helmholtz NTK (`helmholtz-NTK.py`) supports `--optimizer adam` and `--optimizer learnable` only (no quasi-Newton path in that script).

### Optimizers (`--optimizer`)

- **`adam`** — Standard Adam on the network weights in phase 1; collocation and boundary weight variables still use Adam. Phase 2 uses **L-BFGS** on the flattened network weights (same behavior as the original SA-PINN-style scripts).

- **`learnable`** — The **learnable optimizer** (`Optimizers/learnable_optimizer.py`): a small learned network produces update directions for phase 1, while mask weights continue to use Adam. Phase 2 is still **L-BFGS** on the network weights.

- **`quasi-newton`** — *(Allen–Cahn, Burgers, and Helmholtz only.)* Phase 1 uses **Adam** on the network weights (for a stable warm start). Phase 2 uses **`scipy.optimize.minimize`** with the quasi-Newton options used in `Quasi-Newton Optimizer Examples` (e.g. extended BFGS / `method_bfgs` such as `SSBroyden2`). This expects a SciPy installation where `_optimize.py` / `_minimize.py` have been replaced by your modified versions, as in those examples; stock SciPy may ignore some options or behave differently.

### Example commands

Minimal run (defaults: `adam`, 100 / 100 iterations):

```
cd Burgers && python3 burgers.py
```

Learnable phase 1, longer runs:

```
cd "Allen-Cahn" && python3 AC.py --optimizer learnable --tf-iter 5000 --newton-iter 500
```

Quasi-Newton phase 2 (after Adam phase 1):

```
cd Helmholtz && python3 helmholtz.py --optimizer quasi-newton --tf-iter 2000 --newton-iter 20000
```

Helmholtz NTK with custom iteration counts:

```
cd Helmholtz && python3 helmholtz-NTK.py --optimizer adam --tf-iter 200 --newton-iter 200
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
