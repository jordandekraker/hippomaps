#!/usr/bin/env python

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HippoMaps",
    version="0.1.17",
    packages=setuptools.find_packages(),
    include_package_data=True,  # This is key for including package data
    author="Jordan DeKraker",
    author_email="jordandekraker@gmail.com",
    description="A toolbox for viewing, manipulating, and additional actions on HippUnfold outputs",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/jordandekraker/hippomaps",
    package_data={
        'hippomaps': ["resources/*"],
    },
    license="GPL-3.0 license",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords='hippunfold registration',
    python_requires=">=3.7.0",
    install_requires=[
        "Path>=16.4.0",
        "joblib",
        "brainspace>=0.1.2",
        "nibabel>=3.2.2",
        "nilearn>=0.7.0",
        "numpy>=1.23.1,<2.0.0",
        "pandas>=0.23",
        "scipy>=1.3.3",
        "matplotlib>=2.0.0",
        "pygeodesic>=0.1.8",
        "wget>=3.2",
        "eigenstrapping>=0.1",
        "netneurotools==0.2.5",
        "adjusttext>=1.3.0",
	"pathlib",
	"parspin",
	"tabulate",
	"umap-learn>=0.5.5",
    ],
    extras_require={"dev": ["gitpython", "hcp-utils", "mypy", "plotly", "pytest"]},
    zip_safe=False,
)
