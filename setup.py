# -*- coding: utf-8 -*-
import sys

from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

extra_compile_args = ["-O3"]
extra_link_args = []
if sys.platform == "darwin":
    extra_compile_args.append("-mmacosx-version-min=10.9")
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

extensions = [Extension(
    'pydtw.dtw',
    ["pydtw/dtw.pyx"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++")
]

setup(
    name="pydtw",
    description='Fast Implementation of Dynamic Time Warping',
    version="2.0.3",
    long_description=open('README.rst').read(),
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=["numpy"],
    author='Shunsuke Aihara',
    author_email="aihara@argmax.jp",
    url='https://github.com/shunsukeaihara/pydtw',
    license="MIT License",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ]
)
