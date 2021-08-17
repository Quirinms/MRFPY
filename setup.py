import setuptools

from MRFPY import __service_name__, __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name=__service_name__,
    version=__version__,
    author="Quirin Stier",
    author_email="Quirin_Stier@gmx.de",
    description="Timeseries forecasting with wavelets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={'R-Version': 'https://github.com/Quirinms/MRFR', 
                  'Source': 'https://github.com/Quirinms/MRFPY'},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	install_requires=[
          'pandas>=1.0.0',
          'numpy>=1.16',
		  'scipy>=1.1.0',
		  'scikit-learn>=0.21.0'
    ],
    python_requires='>=3.6',
)
