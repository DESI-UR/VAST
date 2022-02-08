
.. role:: raw-html(raw)
    :format: html



############
Introduction
############

:raw-html:`<strong>V<sup>2</sup></strong>` is a software package for finding 
voids based on the ZOBOV (ZOnes Bordering On Voidness) algorithm by 
`Neyrinck (2008) <https://arxiv.org/abs/0712.3049>`_, which uses the gradient of 
the volume of adjacent voronoi cells to flow multiple cells into larger void 
regions.

**VAST** contains the package, as well as an example script for automatically 
running all steps of the algorithm. To import the main void-finding class of 
:raw-html:`<strong>V<sup>2</sup></strong>`::

    from vast.vsquared import zobov

**Voronoi tesselation**

:raw-html:`<strong>V<sup>2</sup></strong>` first produces a Voronoi tessellation 
of the catalog of large-scale tracers, and the volumes of the Voronoi cells are 
used to identify local density minima.

**Zones**

Zones are then built from density minima in the distribution of the Voronoi 
cells using a watershed transform, where each cell is linked to its least dense 
neighbor.

**Defining the voids**

Finally, voids are formed from these by identifying low-density boundaries 
between adjacent zones and using them to grow unions of weakly divided zones.  
This list of voids is then typically pruned to remove void candidates unlikely 
to be true voids.  :raw-html:`<strong>V<sup>2</sup></strong>` includes several 
of the different void-pruning methods that exist, including methods from other 
ZOBOV implementations such as `VIDE <http://www.cosmicvoids.net/>`_ and 
`REVOLVER <https://github.com/seshnadathur/Revolver/>`_.





.. _V2-install:

Installation
============

Operation system requirements
-----------------------------

:raw-html:`<strong>V<sup>2</sup></strong>` is currently Unix-only.


Building V\ :sup:`2`
--------------------

:raw-html:`<strong>V<sup>2</sup></strong>` does not yet have any pre-built 
wheels or distribution packages, so clone the repository from 
https://github.com/DESI-UR/VAST.

:raw-html:`<strong>V<sup>2</sup></strong>` will install like a normal python 
package via the shell command::

    python setup.py install
    
from ``VAST``.  It is important to remember that this will attempt to install 
:raw-html:`<strong>V<sup>2</sup></strong>` into the ``site-packages`` directory 
of whatever python environment that you are using.  To check on this, you can 
type::

    which python
    
into a normal unix shell and it will give you a path like ``/usr/bin/python`` or 
``/opt/anaconda3/bin/python``, which lets you know which python binary your 
``python`` command actually points to.

Developing V\ :sup:`2`
^^^^^^^^^^^^^^^^^^^^^^

If you are actively developing :raw-html:`<strong>V<sup>2</sup></strong>`, you 
can install the package via::

    python setup.py develop
    
which installs a symlink into your python environment's ``site-packages`` 
directory, and the symlink points back to wherever your local copy of the 
:raw-html:`<strong>V<sup>2</sup></strong>` directory is.


Installing V\ :sup:`2` without admin privileges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are working in an environment where you cannot install 
:raw-html:`<strong>V<sup>2</sup></strong>`, or you do not have permissions to 
install it into the python environment that you are using, add ``--user`` to 
your choice of build from above.  For example:: 

    python setup.py develop --user






Citation
========

Please cite `Neyrinck (2008) <https://arxiv.org/abs/0712.3049>`_ when using this 
algorithm.



