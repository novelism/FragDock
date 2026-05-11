# FragDock

FragDock is a framework for fragment-based ligand design using building block assembly and tethered docking.
It provides multiple search methods, including reinforcement learning and baseline search algorithms, within the same molecular assembly and docking workflow.

**Associated manuscript:**  
**FragDockRL: A Reinforcement Learning Method for Fragment-Based Ligand Design via Building Block Assembly and Tethered Docking**  
Seung Hwan Hong et al.  
Preprint available on bioRxiv.

---

## Overview

FragDock explores synthetically accessible chemical space by assembling molecules from building blocks through predefined reaction templates. Generated molecules are evaluated by tethered docking using a reference core structure.

The framework includes the following search methods:

- `FragDockRL`: reinforcement learning-based search
- `random`: random search baseline
- `beam search`: beam search baseline
- `MCTS`: Monte Carlo tree search baseline
- `1-step`: one-step building block search

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-repo>/FragDock.git
cd FragDock
```

---

### 2. Create the conda environment

Create the base environment:

```bash
conda env create -f environment.yml
conda activate fragdock
```

This installs core dependencies such as RDKit, pandas, and docking tools.

---

### 3. Install PyTorch separately

PyTorch is **not included** in `environment.yml` because installation depends on your system, such as CPU/GPU setup and CUDA version.

The following commands are tested with PyTorch 2.10.0.

#### CPU-only

```bash
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### NVIDIA GPU, CUDA 12.8 example

```bash
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

If your system uses a different CUDA version, refer to the official PyTorch installation guide:

```text
https://pytorch.org/get-started/locally/
```

---

### 4. Install FragDock

```bash
pip install .
```

For development, use editable mode:

```bash
pip install -e .
```

---

### 5. Additional dependencies

If AutoDockTools for Python 3 is not already installed:

```bash
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git
```

---

## Usage

After installation, the following command-line scripts are available:

```bash
prepare_core.py
run_fragdock_rl.py
run_fragdock_random.py
run_fragdock_bs.py
run_fragdock_mcts.py
run_fragdock_1step.py
run_tdock.py
```

Each search method is controlled by a YAML configuration file in `configs/`.

```text
configs/
├── f_config_rl.yaml
├── f_config_random.yaml
├── f_config_bs.yaml
├── f_config_mcts.yaml
└── f_config_1step.yaml
```

Typical usage:

```bash
run_fragdock_rl.py -c configs/f_config_rl.yaml
run_fragdock_random.py -c configs/f_config_random.yaml
run_fragdock_bs.py -c configs/f_config_bs.yaml
run_fragdock_mcts.py -c configs/f_config_mcts.yaml
run_fragdock_1step.py -c configs/f_config_1step.yaml
```

To run tethered docking for a prepared ligand/receptor setup, use:

```bash
run_tdock.py <config.yaml>
```

---

## Data preparation

Building block and reaction input files should be prepared before running FragDock.

See:

```text
data/README.md
```

for information on building block files, reaction templates, and preprocessing.

The repository includes template and reaction files such as:

```text
data/building_blocks_template.csv
data/smirks.csv
```

---

## Configuration

Main configuration sections include:

```text
run
 data
 search
 docking
 cutoff
 penalty
```

The reinforcement learning configuration additionally includes:

```text
model
training
```

In `f_config_rl.yaml`, `search.max_step` indicates the maximum number of building block selections before forced termination.

---

## Notes

Tested environment:

- Python 3.12
- RDKit 2023.09.6
- PyTorch 2.10.0

Additional notes:

- Other PyTorch versions may work but are not officially validated.
- `smina` is installed via conda and is required for smina-based docking.
- rDock is required for rDock-based tethered docking.
- PyTorch must be installed separately depending on CPU/GPU setup.

---

## License

Licensed under a Custom Non-Commercial License.
Commercial use requires permission.
