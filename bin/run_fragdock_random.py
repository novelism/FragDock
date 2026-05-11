#!/usr/bin/env python
from fragdock.frag_dock_common import load_config
from fragdock.frag_dock_random import cal_frag_dock_random
import argparse
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.error')


def parse_args():
    parser = argparse.ArgumentParser(description="Run FragDock Random Search")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cal_frag_dock_random(config)


if __name__ == "__main__":
    main()
