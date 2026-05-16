# FA10 Example

This directory provides an example setup for running FragDock on the FA10 target.

## Prepared Input Structures

The `pdb/` directory contains prepared receptor and reference ligand files.

The receptor and co-crystallized ligand should be prepared from the original PDB structure before running FragDock. The receptor should be converted to the formats required by the docking programs, such as MOL2 for rDock and PDBQT for smina.

## Data Files

The `data/` directory is a placeholder for building block and reaction input files.

The actual `building_blocks.csv` file is not distributed with FragDock. Users should prepare this file following the instructions in the main `data/README.md`.

Reaction template files such as `smirks.csv` and `smirks_reactant.csv` can be copied from the main FragDock `data/` directory if needed.

## Docking Setup

The `test/` directory contains target-specific docking setup files.

```text
receptor.prm   rDock receptor/cavity configuration file
config.txt     smina configuration file
```

These files should be prepared in advance for the target protein and binding pocket.

### rDock cavity generation

```bash
rbcavity -was -d -r receptor.prm
```

### Reference core preparation

FragDock requires a reference core conformer for tethered docking.  
The core structure can be extracted from the co-crystallized reference ligand using `prepare_core.py`.

```bash
prepare_core.py "C(=O)c1cccs1" ../pdb/2VWMA_LZI.pdb -c mol_ref_core.pdb
```

If multiple substructure matches are possible, use the `-m` option to select the desired match.

After these steps, the docking setup files and reference core file are ready to be used by the FragDock configuration files in the `test/` directory.

## Running FragDockRL

After preparing the docking setup files and reference core structure, edit the FragDockRL configuration file:

```text
f_config_rl_tdmc.yaml
```

For a simple test run, it is recommended to modify only the starting molecule:

```yaml
start_smi: "OC(=O)c1ccc(Cl)s1"
```

Other options can be left as provided in the default example configuration.

Then run FragDockRL:

```bash
run_fragdock_rl.py -c f_config_rl_tdmc.yaml
```

This example also provides configuration files for other search methods:

```text
f_config_rl_tdmc.yaml    FragDockRL with TD and MC losses
f_config_rl_td.yaml      FragDockRL with TD loss only
f_config_1step.yaml      One-step reaction search
f_config_random.yaml     Random search
f_config_bs.yaml         Beam search
f_config_mcts.yaml       Monte Carlo tree search
```

These configuration files can be used as starting points for testing different FragDock search methods on the same FA10 docking setup.
