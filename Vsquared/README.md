Cosmological paramaters, voidfinding parameters, and input/output files are set in a config file (an example is located in `data/`).  

To find voids and save void and zone information, open a python shell in the `python/` directory and run:

```python
from zobov import Zobov

newZobov = Zobov("/path/to/config.ini",0,3) 
# if intermediate steps 0-3 were saved (the default), use Zobov(4,4) instead; 
# if only 0-n (n<3) were saved, use Zobov(n+1,3) to run the remaining steps

newZobov.sortVoids()
newZobov.saveVoids()
newZobov.saveZones()
```

`saveVoids` produces two output files:
 
a `zobovoids` file with columns for void center positions (`x, y, z, redshift, ra, dec`), void radii (`radius`), and cartesian components of voids' three ellipsoidal axes (`x1, y1, z1, x2, y2, z2, x3, y3, z3`)

and a `zonevoids` file with columns for zone IDs (`zone`), zones' smallest containing void (`void0`), and zones' largest containing void (`void1`).

`saveZones` produces one output file: a `galzones` file with columns for galaxy IDs (`gal`) and for galaxies' containing zone (`zone`).
