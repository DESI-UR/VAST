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
setup_keywords['name'] = 'vast_vsquared'
setup_keywords['description'] = 'Vsquared package'
setup_keywords['author'] = 'Dylan Veyrat, University of Rochester'
setup_keywords['author_email'] = 'dveyrat@ur.rochester.edu'
setup_keywords['license'] = 'BSD 3-clause License'
setup_keywords['url'] = 'https://github.com/DESI-UR/Voids/Vsquared'
setup_keywords['version'] = get_version('vast/vsquared/__init__.py')
requires = []
with open('requirements.txt', 'r') as f:
    for line in f:
        if line.strip():
            requires.append(line.strip())
setup_keywords['install_requires'] = requires
#
# Use README.md as a long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['long_description_content_type'] = 'text/markdown'
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
setup_keywords['test_suite']='tests'
setup_keywords['tests_require']=['pytest']
#
# Run setup command.
#
setup(**setup_keywords)
