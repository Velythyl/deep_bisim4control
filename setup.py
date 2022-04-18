import os
import subprocess

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="redherring",
    version="0.0.1",
    description="Standalone package of distractor environments used in https://github.com/facebookresearch/deep_bisim4control",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Velythyl/distractor-envs",
    author="Charlie Gauthier",
    packages=find_packages(),
    author_email="charlie.gauthier@umontreal.ca",
    license="CC-BY-NC 4.0",
    install_requires=['wheel', "dm_control", "gym", "opencv-python", "scikit-image", "scikit-video", "dm-env", 'matplotlib'],
)