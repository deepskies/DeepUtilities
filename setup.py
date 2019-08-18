from setuptools import setup

setup(
   name='dsutils',
   version='0.1',
   description='Deep Skies Utilities',
   author='Deep Skies @ FNAL',
   author_email='anandj@uchicago.edu',  # todo: change this
   packages=['dsutils'],  #same as name
   install_requires=['torch', 'tensorflow>=2.0.0-beta1', 'pandas', 'numpy', 'h5py', 'matplotlib', 'scikit-learn'], #external packages as dependencies
)
