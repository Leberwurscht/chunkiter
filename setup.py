import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setuptools.setup(
    name="chunkiter",
    version=version,
    author="Leberwurscht",
    author_email="leberwurscht@hoegners.de",
    description="A simple approach and library for doing numpy computations with larger-than-memory data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/leberwurscht/chunkiter",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'tables'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
