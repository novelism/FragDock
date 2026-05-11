#!/usr/bin/env python
import argparse
from fragdock.frag_dock_common import load_config
from fragdock.frag_dock_bs import cal_frag_dock_beam_search
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.error')


def parse_args():
    parser = argparse.ArgumentParser(description="Run FragDock Beam Search.")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    cal_frag_dock_beam_search(config)


if __name__ == "__main__":
    main()
