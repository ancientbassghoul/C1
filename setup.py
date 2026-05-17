from setuptools import setup, find_packages

setup(
    name="raycast_challenge",
    version="0.1.0",
    description="Drone frame spatial alignment and 2D-pixel re-projection pipeline",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8",
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "easyocr>=1.7",
        "Pillow>=10.0",
    ],
)
