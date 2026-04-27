from setuptools import find_packages, setup


setup(
    name="alphapathfold",
    version="0.1.0",
    description="Cleaned AlphaPathFold repository derived from the local Genie source tree",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scipy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
)
