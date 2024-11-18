# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
import os
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="iceshot",
    version="0.1",
    description="Incompressible Cell Shapes via Optimal Transport",  # Required
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://iceshot.readthedocs.io",
    project_urls={
        "Bug Reports": "https://github.com/antoinediez/ICeShOT/issues",
        "Source": "https://github.com/antoinediez/ICeShOT",
    },
    author="Antoine Diez",
    author_email="antoine.n.diez@gmail.com",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="particles shapes optimal transport",
    packages=[
        "iceshot"
    ],
    package_data={
        "iceshot": [
            "readme.md",
            "licence.txt",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "pykeops",
        "tqdm",
        "opencv-python"
    ],
    extras_require={
        "full": [
            "pykeops",
        ],
    },
)