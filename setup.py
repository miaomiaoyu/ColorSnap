from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ColorSnap",
    version="0.1.0",
    author="Miaomiao Yu",
    author_email="mmy@stanford.com",
    description="Extract and visualize dominant color palettes from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miaomiaoyv/colorsnap",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.23.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "colorsnap=colorsnap.colorsnap:main",
        ],
    },
)
