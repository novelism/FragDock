#!/usr/bin/env python
import argparse
from fragdock.frag_dock_common import load_config
from fragdock.frag_dock_1step import cal_frag_dock_1step
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.error')


def parse_args():
    parser = argparse.ArgumentParser(
            description="Run FragDock exhaustive single-step building-block expansion")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cal_frag_dock_1step(config)


if __name__ == "__main__":
    main()
