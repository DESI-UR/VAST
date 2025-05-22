#!/usr/bin/env python
#
# Licensed under a 3-clause BSD style license - see LICENSE.
#
import os
import codecs
from glob import glob
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, dist, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
#
from python.vast._git import get_version
#
# Begin setup
#
setup_keywords = dict()
#
setup_keywords['name'] = 'vast'
setup_keywords['version'] = get_version()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['python_requires'] = '>=3.7'
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages('python')
setup_keywords['package_dir'] = {'': 'python'}
#setup_keywords['package_data'] = {'':['templates/*.glb']}
setup_keywords['test_suite']='vast.test.vast_test_suite.vast_test_suite'
#
# Setup requirements
#
setup_keywords['setup_requires'] = ['Cython', 'numpy']
#
# Setup external Cythonized modules as package extensions.
# Build extensions wrapper to handle numpy includes.
#
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        print(numpy.get_include())
        self.include_dirs.append(numpy.get_include())
#
# Identify all Cython extensions and add them to the extensions list.
#
# The define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] gets rid of the
# annoying numpy API warnings, and the extra_compile_args=["-Wno-maybe-uninitialized"]
# removes all the warnings about using variables which may be uninitialized
ext_modules = []
extfiles = glob('python/vast/voidfinder/*.pyx') + \
           glob('python/vast/voidfinder/*/*.pyx') + \
           glob('python/vast/vsquared/*.pyx')
           
for extfile in extfiles:
    name = extfile.replace('python/', '').replace('/', '.').replace('.pyx', '')
    
    curr_ext = Extension(name, 
                         [extfile], 
                         library_dirs=['m'],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                         extra_compile_args=["-Wno-maybe-uninitialized"])
    
    ext_modules.append(curr_ext)

setup_keywords['ext_modules'] = ext_modules
setup_keywords['cmdclass'] = { 'build_ext' : build_ext }
#
# Run the setup command
#
setup(**setup_keywords)
