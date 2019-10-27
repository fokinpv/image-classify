from setuptools import setup

setup(
    name='image-classify',
    version='0.4.0',
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
    packages=['image_classify'],
    python_requires='>=3.6',
    # scripts=['image-classify']
    entry_points='''
        [console_scripts]
        image-classify=image_classify.cli:cli
    ''',
)
