#!/usr/bin/env python
#
# Licensed under a 3-clause BSD style license - see LICENSE.
#
import os
import codecs
from glob import glob

import re
from os.path import abspath, exists, isdir, isfile, join

#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, dist, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext



# Can't do 'from python.vast._git import get_version' since 'python' is just a 
# folder in our repository, and not a python package itself
# Instead, moving the two utility functions from the _git.py module directly 
# here since this is the only place they are used anyway, they are simple, and 
# Segev wrote nice docstrings.  If we need some more complex get_version() 
# machinery we can revisit this at that time.
# from python.vast._git import get_version

"""Some code for interacting with git.
"""

def get_version():
    """Get the value of ``__version__`` without having to import the package.

    Returns
    -------
    :class:`str`
        The value of ``__version__``.
    """
    ver = 'unknown'
    try:
        version_dir = find_version_directory()
    except IOError:
        return ver

    version_file = join(version_dir, '_version.py')
    with open(version_file, 'r') as f:
        for line in f.readlines():
            mo = re.match("__version__ = '(.*)'", line)
            if mo:
                ver = mo.group(1)
    return ver


def find_version_directory():
    """Return the name of a directory containing version information.
    Looks for files in the following places:
    * python/vast/_version.py
    * vast/_version.py

    Returns
    -------
    :class:`str`
        Name of a directory that can or does contain version information.

    Raises
    ------
    IOError
        If no valid directory can be found.
    """
    packagename = 'vast'
    setup_dir = abspath('.')

    if isdir(join(setup_dir, 'python', packagename)):
        version_dir = join(setup_dir, 'python', packagename)
    elif isdir(join(setup_dir, packagename)):
        version_dir = join(setup_dir, packagename)
    else:
        raise IOError('Could not find a directory containing version information!')
    return version_dir





# Begin setup
#
setup_keywords = dict()
#
setup_keywords['name'] = 'vast'
setup_keywords['version'] = get_version()
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['python_requires'] = '>=3.8'
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
# The define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] gets rid 
# of the annoying numpy API warnings, and the 
# extra_compile_args=["-Wno-maybe-uninitialized"] removes all the warnings about 
# using variables which may be uninitialized
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
