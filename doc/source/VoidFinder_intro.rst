############
Introduction
############

**VAST.VoidFinder** is a software package containing a Python 3 implementation 
of the VoidFinder algorithm 
(`El-Ad & Piran, 1997 <https://arxiv.org/abs/astro-ph/9702135>`_) that is based 
on the algorithm's Fortran implementation by 
`Hoyle & Vogeley (2002) <https://arxiv.org/abs/astro-ph/0109357>`_.  Motivated 
by the expectation that voids are spherical to first order, this algorithm 
defines voids as the unions of sets of spheres grown in the underdense regions 
of the large-scale structure.

The **VoidFinder** directory contains the package, which includes an efficient 
Multi-Process Cythonized version of VoidFinder.  Options are available to 
identify voids in both observational galaxy surveys and periodic cosmological 
simulations.  To import the main void-finding function of VoidFinder::
    
    from vast.voidfinder import find_voids

**VoidFinder** begins by removing all isolated tracers from a catalog of 
objects.  The remaining tracers are then placed on a grid, and spheres are grown 
from the centers of the empty cells until they are bounded by four tracers on 
their surfaces.

All spheres larger than a specified radius (typically around 10 Mpc/h) are 
considered possible maximal spheres -- the largest sphere that can fit in a 
given void region.  Filtering through these candidate maximal spheres by order 
of decreasing radius, no maximal sphere can overlap by more than 10% of its 
volume with any other previously identified (larger) maximal sphere.  After the 
maximal spheres are identified, the remaining holes are combined with these 
maximal spheres to enhance the void structure if they overlap exactly one 
maximal sphere by at least 50% of its volume.  The union of a set of spheres 
(one maximal and the remaining smaller holes) defines a void region.
   




.. _VF-install:

Installation
============

Operating System Support & Requirements
---------------------------------------

*Linux* --- **VoidFinder** is currently only fully supported on Linux, but may run 
on OSX and other Unix variants.

*OSX* --- The authors have successfully run **VoidFinder** with full 
multi-processing power on Mac/OSX, but until Apple guarantees a POSIX- or Single 
Unix Specification-compliant ``fork()`` system call, full multi-processing 
**VoidFinder** cannot be guaranteed.  However, the single-process version of 
**VoidFinder** should always work on OSX to the best of our knowledge, and (as 
of March 2022), **VoidFinder** seems to be working correctly on OSX.

*Windows* --- The authors have encountered difficulty compiling the Cython code 
on the Windows platform, and more importantly the ``fork()`` system call is not 
supported by Windows.  Given its small popularity as a scientific computing 
platform, we have no plans (as of March 2022) to support Windows.  




Building VoidFinder
-------------------

**VoidFinder** does not yet have any pre-built wheels or distribution packages, 
so clone the repository from https://github.com/DESI-UR/VAST.

**VoidFinder** will install like a normal python package via the shell command::

    python setup.py install
    
from ``VAST``.  It is important to remember that this will attempt to install 
`vast.voidfinder` into the ``site-packages`` directory of whatever python 
environment that you are using.  To check on this, you can type::

    which python
    
into a normal unix shell and it will give you a path like ``/usr/bin/python`` or 
``/opt/anaconda3/bin/python``, which lets you know which python binary your 
``python`` command actually points to.


Developing VoidFinder
^^^^^^^^^^^^^^^^^^^^^

If you are actively developing **VoidFinder**, you can install the package via::

    python setup.py develop
    
which installs a symlink into your python environment's ``site-packages`` 
directory, and the symlink points back to wherever your local copy of the 
**VoidFinder** directory is.

If you are developing **VoidFinder** and need to rebuild the cython, from the 
``VAST`` directory run::

    python setup.py build_ext --inplace

Note that you will need to add the path name to where your copy of the **VAST** 
repository lives to your scripts::

    import sys
    sys.path.insert(0, '/path/to/your/VAST/python/')
 
Occasionally, it can be helpful to know the following command::

    cython -a *.pyx
    
which can be run from within the directory where the .pyx files live 
(currently ``VAST/python/vast/voidfinder/``) to "manually" build the cython 
(.pyx) files.


Installing VoidFinder without admin privileges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working in an environment where you cannot install **VoidFinder**, or 
you do not have permissions to install it into the python environment that you 
are using, add ``--user`` to your choice of build from above.  For example:: 

    python setup.py develop --user






Citation
========

Please cite `Hoyle & Vogeley (2002) <https://arxiv.org/abs/astro-ph/0109357>`_ 
and `El-Ad & Piran (1997) <https://arxiv.org/abs/astro-ph/9702135>`_ when using 
this algorithm.




