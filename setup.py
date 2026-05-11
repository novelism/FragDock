from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent

setup(
    name="fragdock",
    version="0.5.0",
    description="Fragment-based molecular generation with reinforcement learning and docking",
    long_description=(this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Seung Hwan Hong",
    packages=find_packages(exclude=["examples*", "data*", "configs*", "bin*"]),
    include_package_data=True,
    scripts=[
        "bin/prepare_core.py",
        "bin/run_fragdock_random.py",
        "bin/run_fragdock_bs.py",
        "bin/run_fragdock_1step.py",
        "bin/run_fragdock_mcts.py",
        "bin/run_fragdock_rl.py",
        "bin/run_tdock.py",
    ],
    python_requires=">=3.12",
)
