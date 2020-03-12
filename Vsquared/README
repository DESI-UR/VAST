Cosmological paramaters can be modified at the top of `python/util.py`, voidfinding parameters and input/output files at the top of `python/zobov.py`.  

To find voids and save void and zone information, open a python shell in the `python/` directory and run:

```python
from zobov import Zobov
newZobov = Zobov(0,3) #if intermediate steps 0-3 were saved (the default), use Zobov(4,4) instead; if only 0-n (n<3) were saved, use Zobov(n+1,3) to run the remaining steps
newZobov.sortVoids()
newZobov.saveVoids()
newZobov.saveZones()
```
