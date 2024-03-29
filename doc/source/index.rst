.. VAST documentation master file, created by
   sphinx-quickstart on Sun Jan 31 18:17:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. role:: raw-html(raw)
    :format: html

Welcome to VAST's documentation!
================================

The **Void Analysis Software Toolkit**, or VAST, provides Python implementations 
of two popular classes of void-finding algorithms in galaxy catalogs:

1. Void identification by growing spherical voids.
2. Void identification using watershed algorithms.
  

Voids are expansive regions in the universe containing significantly fewer 
galaxies than surrounding galaxy clusters and filaments.  They are a fundamental 
feature of the cosmic web and provide important information about galaxy physics 
and cosmology.  For example, correlations between voids and luminous tracers of 
large-scale structure improve constraints on the expansion of the universe as 
compared to using tracers alone.  However, what constitutes a void is vague and 
formulating a concrete definition to use in a void-finding algorithm is not 
trivial.  As a result, several different algorithms exist to identify these 
cosmic underdensities.  VAST provides Python 3 implementations of two such 
algorithms: **VoidFinder** and :raw-html:`<strong>V<sup>2</sup></strong>`.

**VAST** also contains an implementation called **VoidRender** which utilizes OpenGL to
create interactive 3D visualizations of the output of the void-finding algorithms
overlaid on top of the reference galaxy survey.  Each of the void-finding algorithms
in **VAST** provides its own visualization class.






.. toctree::
   :maxdepth: 2
   :caption: VAST
   
   VAST_install
   contributing

.. toctree::
   :maxdepth: 2
   :caption: VoidFinder
   
   VoidFinder_intro
   VoidFinder_examples
   VoidFinder_functions

.. toctree::
   :maxdepth: 2
   :caption: V²
   
   Vsquared_intro
   Vsquared_examples
   Vsquared_functions

.. toctree::
   :maxdepth: 2
   :caption: VoidRender
   
   VoidRender
   VoidRender_examples
   VoidRender_functions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
