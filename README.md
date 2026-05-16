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

Clone the repository:

```bash
git clone https://github.com/novelism/FragDock.git
cd FragDock
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate fragdock
```

Install PyTorch separately according to your CPU/GPU setup and CUDA version:

```text
https://pytorch.org/get-started/locally/
```

The code has been tested with PyTorch 2.10.0.

Install FragDock:

```bash
pip install .
```

For development, use editable mode:

```bash
pip install -e .
```

---

## Data Preparation

Building block and reaction input files should be prepared before running FragDock.

The repository includes template and reaction files such as:

```text
data/building_blocks_template.csv
data/smirks.csv
data/smirks_reactant.csv
```

The actual `building_blocks.csv` file is not distributed with this repository.  
See `data/README.md` for details.

---

## Preparing the Reference Core

FragDock requires a reference core PDB file for tethered docking.  
A core PDB file can be extracted from a reference ligand structure using `prepare_core.py`.

```bash
prepare_core.py "CORE_SMILES" reference_ligand.pdb -c mol_ref_core.pdb
```

Use `prepare_core.py -h` for detailed options.

---

## Docking Setup

FragDock requires target-specific docking setup files for rDock and/or smina.  
Because these settings depend on the protein target and reference ligand, detailed setup instructions will be provided with example cases in the `examples/` directory.

---

## Configuration

Template configuration files are provided in the `configs/` directory.

| Method | Configuration template |
|---|---|
| FragDockRL | `configs/f_config_rl.yaml` |
| One-step search | `configs/f_config_1step.yaml` |
| Random search | `configs/f_config_random.yaml` |
| Beam search | `configs/f_config_bs.yaml` |
| MCTS | `configs/f_config_mcts.yaml` |

Copy the appropriate template file and edit it for your target system.  
See `configs/README.md` for details.

---

## Running FragDock

FragDock provides separate command-line scripts for each search method.

```bash
run_fragdock_rl.py -c configs/f_config_rl.yaml
run_fragdock_1step.py -c configs/f_config_1step.yaml
run_fragdock_random.py -c configs/f_config_random.yaml
run_fragdock_bs.py -c configs/f_config_bs.yaml
run_fragdock_mcts.py -c configs/f_config_mcts.yaml
```

---

## Output Files

Generated molecules, docking results, episode records, and log files are written to the output directory specified in the configuration file.

---

## Notes

Tested environment:

- Python 3.12
- RDKit 2023.09.6
- PyTorch 2.10.0

Additional notes:

- PyTorch must be installed separately depending on the CPU/GPU setup.
- rDock and/or smina input files must be prepared separately for each target system.

---

## License

Licensed under a Custom Non-Commercial License.  
Commercial use requires permission.
