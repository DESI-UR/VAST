# VoidFinder

The VoidFinder algorithm by [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) is based upon the algorithm described by [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E).  It removes all isolated galaxies (described as having the third nearest-neighbor more than ~7 Mpc/h away) using only galaxies with absolute magnitudes M<sub>r</sub> < -20.  After applying a grid to the remaining galaxies, spheres are grown from all empty cells until it is bounded by four galaxies on the surface.  A sphere must have a minimum radius of 10 Mpc/h to be considered part of a void.  If two spheres overlap by more than 10% of their volume, they are considered part of the same void.

## Citation

Please cite [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) and [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E) when using this algorithm.


## Operating System

VoidFinder is currently Unix-only.  VoidFinder relies on the tmpfs filesystem (RAMdisk) on /dev/shm for shared memory, 
and this filesystem is currently (as of February 2020) a Linux-only feature.  However, VoidFinder will fall back to 
memmapping files in the /tmp directory if /dev/shm does not exist, so can still run on OSX.  Depending on the
OSX kernel configuration, there may be no speed/performance loss if running shared memory out of /tmp,
but it entirely depends on the kernel buffer sizes.

Also, VoidFinder uses the fork() method for its worker processes, and the fork() method does
not exist on Windows.

Single & Multi process versions tested successfully on 64-bit Ubuntu 18.04 and 64-bit OSX 10.14.6, with Python 3.7.3.

We have not yet been able to resolve the Windows mishandling of the complex.h library.


## Building Voidfinder

VoidFinder doesn't yet have any pre-built wheels or distribution packages, so clone the repository
from https://github.com/DESI-UR/Voids.git

If you also want to run the VoidRender visualization class which is packaged with VoidFinder, see the additional instructions below under "Building VoidRender"

VoidFinder will install like a normal python package via the shell command 
```
python setup.py install
```
from the `/python/` directory (`Voids/VoidFinder/python/`).  And remember, this will attempt to install VoidFinder into the `site-packages` directory of whatever python environment you're using.  To check on that, in a normal unix shell you can type `which python` and it will give you a path like `/usr/bin/python` or `/opt/anaconda3/bin/python` which lets you know which python binary your `python` command actually points to.


Or, if you're actively developing on VoidFinder, you can do a `python setup.py develop` instead of `python setup.py install` which essentially installs a symlink into your python environment's `site-packages` directory, and the symlink just points back to wherever your local copy of the VoidFinder directory is.


If you're developing VoidFinder and need to rebuild the cython, from the `/python/` directory run:

```
python setup.py build_ext --inplace
```

Occasionally, it can be helpful to know the following command:

```
cython -a *.pyx
```

which can be run from within the directory where the .pyx files live (currently `/python/voidfinder/` and `/python/voidfinder/viz/` to sort of 'manually' cythonize the cython (.pyx) files.  If `python setup.py install` or `python setup.py develop` fails for some reason, try the `cython -a *.pyx` command and then re-try the develop/install command.
 

If you happen to be working in an environment where you can't `install` VoidFinder, or don't have permissions to install it into the python environment you're using, use the above `build_ext --inplace` method, and in your run scripts you can append your local VoidFinder build to the python environment like so:

```
import sys
sys.path.insert(0, "/path/to/your/VoidFinder/voidfinder/python/")
```

The current version of VoidFinder is written to run with Python 3.7.


## Running Voidfinder

There are a few example scripts in the `/python/scripts/` directory of the repository, see `/python/scripts/SDSS_VoidFinder_dr7.py` for an example of running the main algorithm.

Also see `/python/scripts/visualize_voids.py` for an OpenGL-based 3D renderization of the output of the VoidFinder algorithm.

## Building VoidRender (OpenGL-based VoidFinder 3D visualization program)

Requires OpenGL >= 1.2
Requires vispy `pip install vispy`

Example run script in the repository at `/python/scripts/visualize_voids.py`

The video recording capability of VoidRender depents on the `ffmpeg` library
On Ubuntu or similar - `sudo apt install ffmpeg`

Also on Ubuntu you may need to install the following:

`sudo apt-get install mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev`





