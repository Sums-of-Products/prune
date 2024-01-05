from setuptools import setup, find_packages

setup(
    name='prune',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'my-command = main:main'
        ],
    },
)
