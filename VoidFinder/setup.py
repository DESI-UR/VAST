#!/usr/bin/env python
#
# Setup for the voidfinder package. Note that we are using a pkgutil-style
# namespace, so the package name must be vast_voidfinder to match the directory
# structure in this folder.
#
# For details see:
# - https://packaging.python.org/guides/packaging-namespace-packages/
# - https://github.com/pypa/sample-namespace-packages/tree/master/pkgutil
#
# Standard imports.
#
import os
import codecs
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, dist, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

#
# Build file path prefix
# 
# This is to make the ReadTheDocs work
#
cwd = os.getcwd()
requirements_filepath = 'requirements.txt'
working_contents = os.listdir(cwd)

if requirements_filepath in working_contents:
    path_prefix = ''
else:
    # Assumes we are one level up
    path_prefix = 'VoidFinder/'

#
# Version reader
#
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')
#
# Build extensions wrapper to handle numpy includes.
#
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print(numpy.get_include())
        self.include_dirs.append(numpy.get_include())
#
# Setup function.
#
setup(
    name='vast_voidfinder',
    description='VoidFinder package',
    long_description=open(path_prefix + 'README.md').read(),
    author='Kelly Douglass, University of Rochester',
    author_email='kellyadouglass@rochester.edu',
    license='BSD 3-clause License',
    url='https://github.com/DESI-UR/VAST/VoidFinder',
    version=get_version('vast/voidfinder/__init__.py'),

    #packages=['vast.voidfinder',
    #          'vast.voidfinder.viz',
    #          'vast.voidfinder.volume'],
    packages=find_packages(),

    # Requirements.
    requires=['Python (>3.7.0)'],
    install_requires=open(path_prefix + 'requirements.txt', 'r').read().split('\n'),
    zip_safe=False,
    use_2to3=False,

    # Unit tests.
    test_suite='tests',
    tests_require='pytest',

    # Set up cython modules.
    setup_requires=['Cython', 'numpy'],
    ext_modules = [
          Extension('vast.voidfinder._voidfinder_cython_find_next',
                    [path_prefix + 'vast/voidfinder/_voidfinder_cython_find_next.pyx'],
                    library_dirs=['m']),
          Extension('vast.voidfinder._voidfinder_cython',
                    [path_prefix + 'vast/voidfinder/_voidfinder_cython.pyx'],
                    include_dirs=['.'],
                    library_dirs=['m']),
          Extension('vast.voidfinder._vol_cut_cython',
                    [path_prefix + 'vast/voidfinder/_vol_cut_cython.pyx'],
                    include_dirs=['.'],
                    library_dirs=['m']),
          Extension('vast.voidfinder._hole_combine_cython',
                    [path_prefix + 'vast/voidfinder/_hole_combine_cython.pyx'],
                    include_dirs=['.'],
                    library_dirs=['m']),
          Extension('vast.voidfinder.distance',
                    [path_prefix + 'vast/voidfinder/distance.pyx'],
                    include_dirs=['.'],
                    library_dirs=['m']),
          Extension('vast.voidfinder.viz.unionize',
                    [path_prefix + 'vast/voidfinder/viz/unionize.pyx'],
                    library_dirs=['m']),
          Extension('vast.voidfinder.viz.neighborize',
                    [path_prefix + 'vast/voidfinder/viz/neighborize.pyx'],
                    library_dirs=['m']),
          Extension('vast.voidfinder.volume.void_volume',
                    [path_prefix + 'vast/voidfinder/volume/void_volume.pyx'],
                    library_dirs=['m'])
    ],

    cmdclass={'build_ext':build_ext}
)
