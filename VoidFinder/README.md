# VoidFinder

The VoidFinder algorithm by [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) is based upon the algorithm described by [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E).  It removes all isolated galaxies (described as having the third nearest-neighbor more than ~7 Mpc/h away) using only galaxies with absolute magnitudes M<sub>r</sub> < -20.  After applying a grid to the remaining galaxies, spheres are grown from all empty cells until it is bounded by four galaxies on the surface.  A sphere must have a minimum radius of 10 Mpc/h to be considered part of a void.  If two spheres overlap by more than 10% of their volume, they are considered part of the same void.

## Citation

Please cite [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) and [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E) when using this algorithm.


## Operating System

Currently the Multi-Processed version of VoidFinder is Unix-only.  VoidFinder relies on
the tmpfs filesystem (RAMdisk) on /dev/shm for shared memory, and this filesystem is currently
(as of February 2020) a Linux-only feature.  However, VoidFinder will fall back to memmapping
files in the /tmp directory if /dev/shm does not exist, so can still run on OSX.  Depending on the
OSX kernel configuration, there may be no speed/performance loss if running shared memory out of /tmp,
but it entirely depends on the kernel buffer sizes.

Also, VoidFinder uses the fork() method for its worker processes, and the fork() method does
not exist on Windows.

Single & Multi process versions tested successfully on 64-bit Ubuntu 18.04 and 64-bit OSX 10.14.6, with Python 3.7.3.

The single-process version of VoidFinder should run on Linux, OSX, and Windows.


## Building Voidfinder

VoidFinder doesn't yet have any pre-built wheels or distribution packages, so clone the repository
from https://github.com/DESI-UR/Voids.git

VoidFinder will install like a normal python package via `python setup.py install`
from the `/python/` directory (`Voids/VoidFinder/python/`)


Or, if you're working on it, you can do a `python setup.py develop` which essentially
installs a symlink into your python environment which points back to this directory.


If you're developing VoidFinder and need to rebuild the cython, from the `/python/` directory run:

```
python setup.py build_ext --inplace
```

Occasionally, it can be helpful to know the following command:

```
cython -a *.pyx
```

Which can be run from within the directory where the .pyx files live (currently `/python/voidfinder/` and `/python/voidfinder/viz/` to sort of 'manually' cythonize the cython files.


To build voidfinder in the directory where you have it on your machine.  

If you happen to be working in an environment where you can't install VoidFinder, or don't have permissions to install it into the python environment you're using, use the above build-in-place method, and in your run scripts you can append your local VoidFinder build to the python environment like so:

```
import sys
sys.path.insert(0, "/path/to/your/VoidFinder/voidfinder/python/")
```

The current version of VoidFinder is written to run with Python 3.7.


## Running Voidfinder

There are a few example scripts in the `/python/scripts/` directory of the repository, see `/python/scripts/SDSS_VoidFinder_dr7.py` for an example of running the main algorithm.

Also see `/python/scripts/visualize_voids.py` for an OpenGL-based 3D renderization of the output of the VoidFinder algorithm.
