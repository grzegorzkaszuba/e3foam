from setuptools import setup, find_packages

setup(
    name="e3foam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],  # PyTorch is installed separately in Dockerfile
    author="Your Name",
    description="E3 OpenFOAM analysis tools",
    python_requires=">=3.6",
) 