#!/usr/bin/env python
#
# Setup for the vsquared package. Note that we are using a pkgutil-style
# namespace, so the package name must be vast_vsquared to match the directory
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
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
from setuptools.extension import Extension
#
setup_keywords = dict()
setup_keywords['name'] = 'vast_vsquared'
setup_keywords['description'] = 'Vsquared package'
setup_keywords['author'] = 'Dylan Veyrat, University of Rochester'
setup_keywords['author_email'] = 'dveyrat@ur.rochester.edu'
setup_keywords['license'] = 'BSD 3-clause License'
setup_keywords['url'] = 'https://github.com/DESI-UR/Voids/Vsquared'
setup_keywords['version'] = '0.1.0.dev1'
setup_keywords['install_requires'] = ['numpy',
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
setup_keywords['packages'] = ['vast.vsquared',
                              'vast.vsquared.viz']
setup_keywords['cmdclass'] = {'sdist': DistutilsSdist}
setup_keywords['test_suite']='nose2.collector.collector'
setup_keywords['tests_require']=['nose2', 'nose2[coverage_plugin]>=0.6.5']
#
# Run setup command.
#
setup(**setup_keywords)
