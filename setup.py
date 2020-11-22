import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimetry",
    version="0.0.1",
    author="Cyril Zhang",
    author_email="cyril.zhang1@gmail.com",
    description="optimizer tinkering code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyrilzhang/optimetry",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
