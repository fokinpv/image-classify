from setuptools import setup, find_packages

setup(
    name='image-classify',
    version='0.0.1',
    description=(
        'CLI interface to experiment with image classification with PyTorch'
    ),
    install_requires=[
        'click>=7',
        'torch>=1',
        'torchvision>=0.3',
        'pillow>=4.3',
        'numpy>=1.16',
    ],
    include_package_data=True,
    packages=find_packages(),
    python_requires='>=3.6',
    scripts=['image-classify']
)
