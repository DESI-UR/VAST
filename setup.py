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
setup_keywords['description'] = 'Void Analysis Software Toolkit (VAST)'
setup_keywords['author'] = 'Kelly Douglass, University of Rochester'
setup_keywords['author_email'] = 'kellyadouglass@rochester.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/DESI-UR/VAST'
setup_keywords['version'] = get_version()
#
# Use README.md as a long description
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
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print(numpy.get_include())
        self.include_dirs.append(numpy.get_include())
#
# Identify all Cython extensions and add them to the extensions list.
#
ext_modules = []
extfiles = glob('python/vast/voidfinder/*.pyx') + glob('python/vast/voidfinder/*/*.pyx')
for extfile in extfiles:
    name = name = extfile.replace('python/', '').replace('/', '.').replace('.pyx', '')
    ext_modules.append(Extension(name, [extfile], library_dirs=['m']))
#
setup_keywords['ext_modules'] = ext_modules
setup_keywords['cmdclass'] = { 'build_ext' : build_ext }
#
# Package requirements
#
requires = []
with open('requirements.txt', 'r') as f:
    for line in f:
        if line.strip():
            requires.append(line.strip())
setup_keywords['install_requires'] = requires
setup_keywords['extras_require'] = {  # Optional
    'dev': ['pytest', 'pytest-benchmark'],
    'docs': ['numpydoc', 'sphinx-argparse', 'sphinx_rtd_theme']
}
#
# Run the setup command
#
setup(**setup_keywords)
