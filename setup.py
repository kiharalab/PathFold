from setuptools import find_packages, setup


setup(
    name="pathfold",
    version="0.1.0",
    description="PathFold protein folding pathway inference repository",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "dm-tree",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
)
