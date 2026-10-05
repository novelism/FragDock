
# FragDock
Current public version: **v0.5**

FragDock is a molecular design framework that explores synthetically accessible chemical space by assembling molecules from building blocks through predefined reaction templates and evaluating them using tethered docking.

## Related Papers

For details of the FragDock framework and FragDockRL method, please refer to the following paper.  
If you use FragDock in your research, please cite this work:

**FragDockRL: A Reinforcement Learning Method for Fragment-Based Ligand Design via Building-Block Assembly and Tethered Docking**  
Seung Hwan Hong et al.  
*Journal of Chemical Information and Modeling*  
https://doi.org/10.1021/acs.jcim.6c02851

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

FragDock requires target-specific docking setup files for rDock and SMINA.
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

Generated molecules, docking results, episode records, training logs, and method-specific result files are written to the output paths specified in the configuration file.

---

## Notes

Tested environment:

- Python 3.12
- RDKit 2023.09.6
- PyTorch 2.10.0

Additional notes:

- PyTorch must be installed separately depending on the CPU/GPU setup.
- rDock and SMINA input files must be prepared separately for each target system.

---

## Research Collaboration and Feedback

We are looking for research partners interested in applying FragDock to real-world drug discovery projects.

Potential collaborations may include:
- target-specific virtual screening
- fragment or hit expansion
- structure-guided molecular design
- experimental validation of FragDock-generated candidates
- development and evaluation of new FragDock workflows

We also welcome feedback and suggestions for improving FragDock, including:
- new features or workflow ideas
- support for additional reaction types
- docking or scoring improvements
- usability and documentation improvements
- bug reports and reproducibility issues

For research collaboration inquiries, contact:  
Seung Hwan Hong  
shhong@novelismlab.com

For bug reports and feature suggestions, please use GitHub Issues.

---

## License

Licensed under a Custom Non-Commercial License.  
Academic and non-commercial use is permitted under the terms of the license.  
Commercial use requires prior permission from the author.

For commercial licensing inquiries, contact:  
Seung Hwan Hong  
shhong@novelismlab.com

