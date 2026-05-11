#!/usr/bin/env python
import argparse
from fragdock.frag_dock_common import load_config
from fragdock.frag_dock_mcts import cal_frag_dock_mcts
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.error')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FragDock Monte Carlo Tree Search")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cal_frag_dock_mcts(config)


if __name__ == "__main__":
    main()
