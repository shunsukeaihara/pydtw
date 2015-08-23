# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import sys
import numpy as np
sys.path.append('./src')
sys.path.append('./test')

ext_modules = [Extension(
        'pydtw.dtw',
        ["pydtw/dtw.pyx"],
        include_dirs=['lib/world', np.get_include()],
        extra_compile_args=["-O3"],
        language="c++")
]

setup(
    name="pydtw",
    description='Fast Imprementation of the Dynamic Wime Warping',
    version="1.0.0",
    long_description=open('README.rst').read(),
    packages=find_packages(),
    install_requires=["numpy", 'nose', 'cython'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    author='Shunsuke Aihara',
    author_email="aihara@argmax.jp",
    url='https://github.com/shunsukeaihara/pydtw',
    license="MIT License",
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose', 'numpy', 'cython'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
    ]
)
