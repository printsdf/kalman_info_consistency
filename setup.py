from setuptools import setup, find_packages

setup(
    name='kalman_info_consistency',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
    ],
)