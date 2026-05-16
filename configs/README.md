# Configuration Files

This directory contains template configuration files for running FragDock with different search methods.

```text
f_config_rl.yaml       FragDockRL
f_config_1step.yaml    One-step search
f_config_random.yaml   Random search
f_config_bs.yaml       Beam search
f_config_mcts.yaml     Monte Carlo tree search
```

## Recommended Usage

The configuration files in this directory are templates.  
Users should copy the required template file to their own project directory and edit the copied file, rather than modifying the original template in the FragDock repository.

Example:

```bash
cp /path/to/FragDock/configs/f_config_rl.yaml /path/to/my_project/config_rl.yaml
```

Then edit the copied file:

```text
/path/to/my_project/config_rl.yaml
```

and run FragDock with:

```bash
run_fragdock_rl.py -c /path/to/my_project/config_rl.yaml
```

The organization of the project directory is left to the user.  
All input, output, docking, and data paths should be set explicitly in the copied configuration file.

## What to Modify

The template files already contain the available options.  
In most cases, users should update:

```text
- input file paths
- output directory and file names
- docking-related paths and settings
- search parameters for the selected method
- model and training parameters for FragDockRL
```

Target-specific docking setup files, such as rDock and smina input files, should be prepared separately.  
Example configurations with prepared docking settings will be provided in the `examples/` directory.
