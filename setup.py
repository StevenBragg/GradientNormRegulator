import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gnr",
    version="0.1.0",
    author="Steven Bragg",
    author_email="your.email@example.com",
    description="A dynamic learning rate adjuster based on gradient norm for machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StevenBragg/GradientNormRegulator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow>=2.4.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "scikit-optimize>=0.8.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9.0",
            "black>=21.5b1",
            "isort>=5.8.0",
            "mypy>=0.812",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/StevenBragg/GradientNormRegulator/issues",
    },
)
