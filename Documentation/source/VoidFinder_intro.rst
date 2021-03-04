############
Introduction
############

`vast.voidfinder` is a software package containing a Python 3 implementation of 
the VoidFinder algorithm [@El-Ad:1997] that is based on the algorithm's Fortran 
implementation by @Hoyle:2002.  Motivated by the expectation that voids are 
spherical to first order, this algorithm defines voids as the unions of sets of 
spheres grown in the underdense regions of the large-scale structure.

The `VoidFinder` directory contains the package, which includes an efficient 
Multi-Process Cythonized version of VoidFinder.  To import the main 
void-finding function of VoidFinder::
    
    from vast.voidfinder import find_voids

`VoidFinder` begins by removing all isolated tracers from a catalog of objects.  
The remaining tracers are then placed on a grid, and spheres are grown from the 
centers of the empty cells until they are bounded by four tracers on their 
surfaces.

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

Operation system requirements
-----------------------------

`VoidFinder` is currently Unix-only.  `VoidFinder` relies on the tmpfs file 
system (RAMdisk) on `/dev/shm` for shared memory, and this file system is 
currently (as of February 2020) a Linux-only feature.  However, `VoidFinder` 
will fall back to memmapping files in the `/tmp` directory if `/dev/shm` does 
not exist, so it can still run on OSX.  Depending on the OSX kernal 
configuration, there may be no speed/performance loss if running shared memory 
out of `/tmp`, but it depends entirely on the kernal buffer sizes.

`VoidFinder` also uses the ``fork()`` method for its worker processes, and the 
``fork()`` method does not exist on Windows.


Building VoidFinder
-------------------

`VoidFinder` does not yet have any pre-built wheels or distribution packages, so 
clone the repository from https://github.com/DESI-UR/VAST.

`VoidFinder` will install like a normal python package via the shell command::

    python setup.py install
    
from `VAST/VoidFinder`.  It is important to remember that this will attempt to 
install `VoidFinder` into the `site-packages` directory of whatever python 
environment that you are using.  To check on this, you can type::

    which python
    
into a normal unix shell and it will give you a path like `/usr/bin/python` or 
`/opt/anaconda3/bin/python`, which lets you know which python binary your 
`python` command actually points to.

Developing VoidFinder
^^^^^^^^^^^^^^^^^^^^^

If you are actively developing `VoidFinder`, you can install the package via::

    python setup.py develop
    
which installs a symlink into your python environment's `site-packages` 
directory, and the symlink points back to wherever your local copy of the 
`VoidFinder` directory is.

If you are developing `VoidFinder` and need to rebuild the cython, from the 
`VAST/VoidFinder` directory run::

    python setup.py build_ext --inplace

Note that you will need to add the path name to where your copy of the `VAST` 
repository lives to your scripts::

    import sys
    sys.path.insert(0, '/path/to/your/VAST/VoidFinder/vast/voidfinder/')
 
Occasionally, it can be helpful to know the following command::

    cython -a *.pyx
    
which can be run from within the directory where the .pyx files live 
(currently `VAST/VoidFinder/vast/voidfinder/`) to "manually" build the cython 
(.pyx) files.

Installing VoidFinder without admin privileges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are working in an environment where you cannot install `VoidFinder`, or 
you do not have permissions to install it into the python environment that you 
are using, add ``--user`` to your choice of build from above.  For example:: 

    python setup.py develop --user






Citation
========

Please cite [@Hoyle:2002] and [@El-Ad:1997] when using this algorithm.




