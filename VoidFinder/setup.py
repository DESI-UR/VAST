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
#
from setuptools import setup, find_packages
#
from vast.voidfinder._git import get_version, SetVersion
#
setup_keywords = dict()
setup_keywords['name'] = 'vast_voidfinder'
setup_keywords['description'] = 'VoidFinder package'
setup_keywords['author'] = 'Kelly Douglass, University of Rochester'
setup_keywords['author_email'] = 'kellyadouglass@rochester.edu'
setup_keywords['license'] = 'BSD 3-clause License'
setup_keywords['url'] = 'https://github.com/DESI-UR/Voids/VoidFinder'
setup_keywords['version'] = get_version()
setup_keywords['install_requires'] = ['cython',
                                      'h5py',
                                      'psutil',
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
setup_keywords['packages'] = ['vast.voidfinder',]
                              #'voidfinder.viz',
                              #'voidfinder.volume']
setup_keywords['test_suite']='nose2.collector.collector'
setup_keywords['tests_require']=['nose2', 'nose2[coverage_plugin]>=0.6.5']
#
# Run setup command.
#
setup(**setup_keywords)
