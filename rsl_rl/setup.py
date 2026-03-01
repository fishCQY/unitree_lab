from setuptools import setup

setup(
    name="rsl-rl-lib",
    version="3.0.1",
    author="unitree_lab",
    description="Custom RSL-RL library with AMP support for unitree_lab",
    packages=[
        "rsl_rl",
        "rsl_rl.algorithms",
        "rsl_rl.env",
        "rsl_rl.modules",
        "rsl_rl.networks",
        "rsl_rl.runners",
        "rsl_rl.storage",
        "rsl_rl.utils",
    ],
    package_dir={"rsl_rl": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "tensordict",
    ],
)
