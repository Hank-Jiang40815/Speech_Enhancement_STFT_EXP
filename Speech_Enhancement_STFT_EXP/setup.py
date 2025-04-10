from setuptools import setup, find_packages

setup(
    name="speech_enhancement",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "numpy>=1.21.0",
        "librosa>=0.8.1",
        "matplotlib>=3.4.0",
    ],
)