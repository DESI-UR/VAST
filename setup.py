#!/usr/bin/env python
#
# Future-proof for Python 2.7 users.
from __future__ import absolute_import, division, print_function
#
# Standard imports.
#
from glob import glob
import os
import re
import sys
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
#
from python.voidfinder._git import get_version, SetVersion
#
# Package setup.
#
setup_keywords = dict()
#
setup_keywords['name'] = 'voidfinder'
setup_keywords['description'] = 'VoidFinder package'
setup_keywords['author'] = 'Kelly Douglass, University of Rochester'
setup_keywords['author_email'] = 'kellyadouglass@rochester.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/DESI-UR/VoidFinder'
setup_keywords['version'] = get_version()
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
setup_keywords['requires'] = ['Python (>2.7.0)']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages('python')
setup_keywords['package_dir'] = {'': 'python'}
setup_keywords['cmdclass'] = {'version': SetVersion, 'sdist': DistutilsSdist}
setup_keywords['test_suite']='nose.collector'
setup_keywords['tests_require']=['nose']
#
# Internal data directories.
#
setup_keywords['data_files'] = [('VoidFinder/data/config', glob('data/config/*')),
                                ('VoidFinder/data/examples', glob('data/examples/*'))]
#
# Run setup command.
#
setup(**setup_keywords)
