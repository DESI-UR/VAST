

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

extensions = [
    
    
    
    
    Extension(
        "dist_funcs_cython",
        ["dist_funcs_cython.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        ),
    
    Extension(
        "_voidfinder_cython",
        ["_voidfinder_cython.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        ),
    Extension(
        "_voidfinder_cython_find_next",
        ["_voidfinder_cython_find_next.pyx"],
        include_dirs=[numpy.get_include()+"/numpy"],
        libraries=["m"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
        
        
        )
              
]

setup(
    name = 'voidfinder',
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