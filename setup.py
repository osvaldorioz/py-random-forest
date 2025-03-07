from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#python3.12 setup.py build_ext --inplace

eigen_include_dir = '/usr/include/eigen3'

setup(
    name='rfe_module',
    ext_modules=[
        CppExtension(
            name='rfe_module',
            sources=['rfe.cpp'],
            include_dirs=[eigen_include_dir],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
