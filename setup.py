import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

#with open("requirements.txt") as file:
#    required = file.read().splitlines()
    
setuptools.setup(
    name="bikit-models",
    version="0.0.1",
    author="Philipp J. Roesch, Johannes Flotzinger",
    author_email="philipp.roesch@unibw.de, johannes.flotzinger@unibw.de",
    description="Models for Building Inspection Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phiyodr/bikit-models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.csv", "*.json"]},
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
        "pathlib",
        "matplotlib",
        "efficientnet_pytorch",
        ]
    )