from setuptools import setup, find_packages

setup(
    name='inductive_clusterer',
    version='0.1.0',
    url='https://github.com/querbesd/public_code',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn', 
        'scipy'
    ],
)
