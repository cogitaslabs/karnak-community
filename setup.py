import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="karnak",
    version="0.1.0",
    author="Leonardo Rossi",
    author_email="leorossi@cogitaslabs.com",
    description="Karnak Data Platform Community Libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogitaslabs/karnak-community",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)