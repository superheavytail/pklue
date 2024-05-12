from setuptools import setup, find_packages

setup(
    name="pklue",
    version="1.0.0",
    description="Korean Datasets for Instruction Tuning",
    packages=find_packages(),
    author="Jeongwook Kim",
    author_email="k0s1k0s1k0@korea.ac.kr",
    install_requires=[
        'datasets',
    ],
)