# -*- coding: utf-8 -*-
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3"]
    extra_link_args = []
    if sys.platform == "darwin":
        extra_compile_args.append("-mmacosx-version-min=10.9")
        extra_compile_args.append("-stdlib=libc++")
        extra_link_args.append("-stdlib=libc++")

extensions = [Extension(
    "pydtw.dtw",
    ["pydtw/dtw.pyx", "pydtw/simd/dtw_simd.cpp"],
    depends=["pydtw/simd/dtw_simd.hpp", "pydtw/simd/kernels.inl"],
    include_dirs=["pydtw"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
