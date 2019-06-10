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
        'pillow>=6',
        'numpy>=1.16',
    ],
    include_package_data=True,
    packages=find_packages(),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['image-classify=image_classify.cli:cli']
    },
)