from setuptools import setup, find_packages

setup(
    name="sb_sched",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pykitti>=0.3.1",
        "rosbags>=0.9",
        "tqdm>=4.65",
        "pandas>=2.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.11",
)
