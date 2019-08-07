import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imagegraph",
    version="0.0.1",
    author="Leonardo Impett",
    author_email="leoimpett@gmail.com",
    description="The tiny Python library required to run code auto-generated on www.imagegraph.cc",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leoimpett/pyimagegraph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)