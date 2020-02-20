# VoidFinder

The VoidFinder algorithm by [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) is based upon the algorithm described by [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E).  It removes all isolated galaxies (described as having the third nearest-neighbor more than ~7 Mpc/h away) using only galaxies with absolute magnitudes M<sub>r</sub> < -20.  After applying a grid to the remaining galaxies, spheres are grown from all empty cells until it is bounded by four galaxies on the surface.  A sphere must have a minimum radius of 10 Mpc/h to be considered part of a void.  If two spheres overlap by more than 10% of their volume, they are considered part of the same void.

## Citation

Please cite [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) and [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E) when using this algorithm.


## Operating System

Currently the Multi-Processed version of VoidFinder is Linux-only.  VoidFinder relies on
the tmpfs filesystem (RAMdisk) on /dev/shm for shared memory which isn't available on OSX or Windows.
Also, the fork() method for spawning workers does not exist on Windows and does not work
correctly on Mac/OSX (it's an Apple/OSX problem, not a Python problem).

The single-process version of VoidFinder should run on Linux, OSX, and Windows.


## Building & Running Voidfinder

VoidFinder will install like a normal python package via `python setup.py install`
from the `/python/` directory (`Voids/VoidFinder/python/`)


Or, if you're working on it, you can do a `python setup.py develop` which essentially
installs a symlink into your python environment which points back to this directory.


If you're developing VoidFinder and need to rebuild the cython, enter the `/python/voidfinder/` directory and run:

```
cython -a *.pyx
```

Then cd back up to the `/python/` directory and use the setup.py script like so:

```
python setup.py build_ext --inplace
```

To build voidfinder in the directory where you have it on your machine.  

If you happen to be working in an environment where you can't install VoidFinder, or don't have permissions to install it into the python environment you're using, use the above build-in-place method, and in your run scripts you can append your local VoidFinder build to the python environment like so:

```
import sys
sys.path.insert(0, "/path/to/your/VoidFinder/voidfinder/python/")
```



The current version of VoidFinder is written to run with Python 3.7.
