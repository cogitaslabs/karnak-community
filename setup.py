import setuptools
import pathlib
import pkg_resources

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


EXTRAS_REQUIRE_AWS = ['boto3', 'redshift_connector', 'PyAthena>=2.1.0', 'PyAthenaJDBC[SQLAlchemy]']
EXTRAS_REQUIRE_GCP = []
EXTRAS_REQUIRE_FULL = EXTRAS_REQUIRE_AWS + EXTRAS_REQUIRE_GCP


def requirements():
    with pathlib.Path('requirements.txt').open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires


setuptools.setup(
    name="karnak",
    version="3.0.2",
    author="Leonardo Rossi",
    author_email="leorossi@cogitaslabs.com",
    description="Karnak Data Platform Community Edition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogitaslabs/karnak-community",
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements(),
    extras_require={
        'aws': EXTRAS_REQUIRE_AWS,
        'gcp': EXTRAS_REQUIRE_GCP,
        'full': EXTRAS_REQUIRE_FULL
    }
)
