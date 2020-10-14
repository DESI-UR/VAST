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
from setuptools import setup, find_packages
from setuptools.extension import Extension
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
# Setup keywords
#
setup_keywords = dict()
setup_keywords['name'] = 'vast_voidfinder'
setup_keywords['description'] = 'VoidFinder package'
setup_keywords['author'] = 'Kelly Douglass, University of Rochester'
setup_keywords['author_email'] = 'kellyadouglass@rochester.edu'
setup_keywords['license'] = 'BSD 3-clause License'
setup_keywords['url'] = 'https://github.com/DESI-UR/Voids/VoidFinder'
setup_keywords['version'] = get_version('vast/voidfinder/__init__.py')
setup_keywords['install_requires'] = ['cython',
                                      'h5py',
                                      'psutil',
                                      'numpy',
                                      'scikit-learn']
#
# Use README.md as a long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.
#
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>3.7.0)']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = ['vast.voidfinder',
                              'vast.voidfinder.viz',
                              'vast.voidfinder.volume']
#setup_keywords['cmdclass'] = {'version': SetVersion, 'sdist': DistutilsSdist}
setup_keywords['test_suite']='nose2.collector.collector'
setup_keywords['tests_require']=['nose2', 'nose2[coverage_plugin]>=0.6.5']
#
# Set up cython build.
#
from Cython.Build import cythonize
import numpy
extensions = [
              Extension("vast.voidfinder._voidfinder_cython_find_next",
                        ["vast/voidfinder/_voidfinder_cython_find_next.pyx"],
                        include_dirs=[".", numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder._voidfinder_cython",
                        ["vast/voidfinder/_voidfinder_cython.pyx"],
                        include_dirs=[".", numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder._vol_cut_cython",
                        ["vast/voidfinder/_vol_cut_cython.pyx"],
                        include_dirs=[".", numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder._hole_combine_cython",
                        ["vast/voidfinder/_hole_combine_cython.pyx"],
                        include_dirs=[".", numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder.distance",
                        ["vast/voidfinder/distance.pyx"],
                        include_dirs=[".", numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder.viz.unionize",
                        ["vast/voidfinder/viz/unionize.pyx"],
                        include_dirs=[numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder.viz.neighborize",
                        ["vast/voidfinder/viz/neighborize.pyx"],
                        include_dirs=[numpy.get_include()],
                        library_dirs=["m"]),
              Extension("vast.voidfinder.volume.void_volume",
                        ["vast/voidfinder/volume/void_volume.pyx"],
                        include_dirs=[numpy.get_include()],
                        library_dirs=["m"])
              ]
#
setup_keywords["ext_modules"] = cythonize(extensions)
#
# Run setup command.
#
setup(**setup_keywords)
