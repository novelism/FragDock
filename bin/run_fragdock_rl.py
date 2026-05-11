#!/usr/bin/env python
import argparse
from fragdock.frag_dock_rl import (get_device, cal_frag_dock_rl,
                                   build_network_and_optimizer)
from fragdock.frag_dock_common import load_config

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')


def parse_args():
    parser = argparse.ArgumentParser(description="Run FragDockRL training.")
    parser.add_argument('-c', "--config", type=str, required=True,
                        help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = get_device(config["run"])
    print(device)

    online_net, target_net, optimizer, loss_function = (
        build_network_and_optimizer(config["model"], device)
    )

    cal_frag_dock_rl(
        config,
        device,
        online_net,
        target_net,
        optimizer,
        loss_function,
    )


if __name__ == "__main__":
    main()
