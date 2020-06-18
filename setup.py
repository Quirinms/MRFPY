import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Multiresolution_Forecasting-(MRF)",
    version="1.0.0",
    author="Quirin Stier",
    author_email="Quirin_Stier@gmx.de",
    description="Timeseries forecasting with wavelets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="None",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
