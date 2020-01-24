

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

extensions = [
    
    Extension(
        "unionize",
        ["unionize.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    
    Extension(
        "neighborize",
        ["neighborize.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        )
              
]

setup(
    name = 'voidfinder-viz',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
    include_dirs=[numpy.get_include(), numpy.get_include()+"/numpy"]
  
)


'''
    Extension(
        "lgamma",
        ["lgamma.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    
    Extension(
        "typedefs",
        ["typedefs.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    
    Extension(
        "dist_metrics",
        ["dist_metrics.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    
    Extension(
        "kd_tree",
        ["kd_tree.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    
    
'''