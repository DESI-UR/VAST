# V<sup>2</sup>

## Description

The V<sup>2</sup> package computes void regions using Voronoi tesselations of a
galaxy catalog using the ZOnes Bordering on Voids (ZOBOV) algorithm developed
by [Mark Neyrinck (2007)](https://doi.org/10.1111/j.1365-2966.2008.13180.x).

## Installation

The package can be installed using setuptools via the shell command
```
python setup.py install
```
from the same directory containing this README file (`VAST/Vsquared`). This
will install V<sup>2</sup> into the `site-packages` folder of your current
Python environment.

If you are actively developing V<sup>2</sup> you can run `python setup.py
develop` to create a symlink from the source tree in your `site-packages`
directory.

## Usage (Python shell)

Cosmological paramaters, voidfinding parameters, and input/output files are set in a config file (an example is located in `Vsquared/data/`).  

To find voids and save void and zone information, open a python shell in the `Vsquared/` directory and run:

```python
from vast.vsquared.zobov import Zobov

newZobov = Zobov("/path/to/config.ini",0,3) # options: save_intermediate (default True), visualize (default False)
# if intermediate steps 0-3 were saved (the default), use Zobov(4,4) instead; 
# if only 0-n (n<3) were saved, use Zobov(n+1,3) to run the remaining steps

newZobov.sortVoids() # pass void pruning method number to this function, default 0
newZobov.saveVoids()
newZobov.saveZones()
# finally, if intending to run visualization:
# newZobov.preViz()
```

`saveVoids` produces two output files:
 
a `zobovoids` file with columns for void center positions (`x, y, z, redshift, ra, dec`), void radii (`radius`), and cartesian components of voids' three ellipsoidal axes (`x1, y1, z1, x2, y2, z2, x3, y3, z3`)

and a `zonevoids` file with columns for zone IDs (`zone`), zones' smallest containing void (`void0`), and zones' largest containing void (`void1`).

`saveZones` produces one output file: a `galzones` file with columns for galaxy IDs (`gal`), galaxies' containing zone (`zone`), and galaxies' "depth" within their containing zone (`depth`).

## Usage (autorun script)

The `Vsquared/data` directory contains `vsquared.py`, a script for automatically running the algorithm with given parameters. Four requirements can be passed to this script, one of which (the config file) is required:

`--config /path/to/config.ini` or `-c /path/to/config.ini`

and three of which are optional:

`--method 0` or `-m 0` (specify pruning method from 0 to 4, default 0)

`--visualize` or `-v` (enable void visualization)

`--save_intermediate` or `-w` (save intermediate files in void calculation)
