from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

lib = []
lib.extend([]);
lib.extend([]);

extensions = [
    Extension("*", ["*.pyx"],
              include_dirs = ["../../src"],
              language='c',
              libraries=lib,
              extra_objects = ["../../src/libfreesasa.a"],
              extra_compile_args = ["-w"] 
              )
]

setup(
    name='FreeSASA',
    description='Calculate solvent accessible surface areas of proteins',
    version= '2.0',
    author='Simon Mitternacht',
    url='http://freesasa.github.io/',
    license='MIT',
    ext_modules = cythonize(extensions)
)
