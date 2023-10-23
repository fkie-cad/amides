from setuptools import setup, find_packages


with open("requirements.in", encoding="utf-8", mode="r") as f:
    requirements = f.read().splitlines()

setup(
    name="amides",
    version="0.1",
    description="Amides package contains proof-of-concept implementation of the Adaptive Misuse Detection System (AMIDES).",
    url="https://github.com/fkie-cad/amides",
    license="GPL-3.0 license",
    packages=find_packages(),
    install_requires=["setuptools"] + requirements,
    python_requires=">=3.10",
)
