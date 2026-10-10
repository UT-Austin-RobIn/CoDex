import os

import setuptools


def _read_requirements():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


setuptools.setup(
    name="codex",
    version="0.1.0",
    author="CoDex authors",
    author_email="jbw0008@outlook.com",
    description="CoDex: dexterous functional object manipulation without demonstrations",
    license="MIT",
    long_description="See README.MD and https://robin-lab.cs.utexas.edu/CoDex/",
    long_description_content_type="text/markdown",
    url="https://robin-lab.cs.utexas.edu/CoDex/",
    # Restrict discovery to the actual source tree. Other top-level folders
    # contain examples, generated outputs, and locally cloned dependencies.
    packages=setuptools.find_packages(include=["codex", "codex.*"]),
    include_package_data=True,
    install_requires=_read_requirements(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
