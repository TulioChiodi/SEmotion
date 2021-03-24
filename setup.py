from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    install_requires=[
        "pip",
    ],
    description='Speech Emotion Recognition using YAMNet',
    author='Tulio Chiodi',
    license='MIT',
)
