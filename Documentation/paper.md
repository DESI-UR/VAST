---
title: 'VoidFinder & V<sup>2</sup> - Void-finding algorithms in Python 3'
tags:
  - Python
  - astronomy
  - voids
  - large-scale structure
  - redshift survey
authors:
  - name: Kelly A. Douglass
    orcid: 0000-0002-9540-546X
    affiliation: 1
  - name: Dylan Veyrat
    orcid: 0000-0001-8101-2836
    affiliation: 1
  - name: Stephen W. O'Neill, Jr.
  - name: Segev BenZvi
    orcid: 0000-0001-5537-4710
    affiliation: 1
affiliations:
  - name: University of Rochester
    index: 1
date: June 2020
bibliography: paper.bib
---


# Summary

Voids are expansive regions in the universe containing significantly fewer galaxies than surrounding galaxy clusters and filaments. They are a fundamental feature of the cosmic web, and provide important information about galaxy physics and cosmology. For example, correlations between voids and luminous tracers of large-scale structure improve constraints on the expansion of the universe compared to using tracers alone. However, the basic concept of a void is vague and formulating a concrete definition to use in a void-finding algorithm is not trivial. As a result, several different algorithms exist to identify these cosmic underdensities, two of which are included in this repository as the bases for the `VoidFinder` and `Vsquared` packages.

Analyzing the next generation of cosmological surveys will require the ability to process large volumes of data (at least 25 million galaxies and quasars from DESI [@DESI]). [in progress]




# VoidFinder







# V<sup>2</sup>

`Vsquared` is a software package for finding voids based on the ZOBOV (ZOnes Bordering On Voidness) algorithm [@Neyrinck:2007]. This algorithm first produces a Voronoi tessellation of the galaxy catalog, and the volumes of the Voronoi cells are used to identify local density minima. Zones are then grown from density minima in the distribution of cells using a watershed transform, where each cell is linked to its least dense neighbor. Finally, voids are formed from these by identifying low-density boundaries between adjacent zones and using them to grow unions of weakly divided zones. A void-pruning step is usually also run at the end of the algorithm, to remove void candidates unlikely to be true voids; there exists a variety of implementations of this step, and `Vsquared' includes several.




# Acknowledgements

We acknowledge support from the DOE Office of High Energy Physics under grant number DE-SC0008475.




# References
