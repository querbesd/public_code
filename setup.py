from setuptools import setup, find_packages

setup(
    name='inductive_clusterer',
    version='0.1.0',
    url='https://github.com/querbesd/public_code',
    packages=find_packages(),
    install_requires=[
        'numpy==1.18.5',
        'scikit-learn==1.0.2',
        'joblib==0.15.1',
        'scipy==1.4.1'
    ],
)
