from setuptools import setup, find_packages


setup(
    name='text-dataset',
    version='0.1',
    packages=find_packages(),
    license=license,
    long_description_content_type="text/markdown",
    long_description="",
    author="Gabriel de Miranda ",
    url="https://github.com/ograndoptimist/TokenizerPlusTorchTensors",
    description="A Pipeline for text data",
    setup_requires=['wheel', 'twine'],
    install_requires=['pandas', 'torch']
)
