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
    affiliation: 2
  - name: Segev BenZvi
    orcid: 0000-0001-5537-4710
    affiliation: 1
affiliations:
  - name: University of Rochester
    index: 1
  - name: Independent Researcher
    index: 2
date: August 2020
bibliography: paper.bib
---


# Summary

Voids are expansive regions in the universe containing significantly fewer galaxies than surrounding galaxy clusters and filaments. They are a fundamental feature of the cosmic web, and provide important information about galaxy physics and cosmology. For example, correlations between voids and luminous tracers of large-scale structure improve constraints on the expansion of the universe compared to using tracers alone. However, the basic concept of a void is vague and formulating a concrete definition to use in a void-finding algorithm is not trivial. As a result, several different algorithms exist to identify these cosmic underdensities. The Void Analysis Software Toolkit, or VAST, provides Python 3 implementations of two such algorithms in the `VoidFinder` and `Vsquared` packages.

Analyzing the next generation of cosmological surveys will require the ability to process large volumes of data (at least 25 million galaxies and quasars from the Dark Energy Spectroscopic Instrument [@DESI]). To more efficiently process such large datasets, the `VoidFinder` package includes a Cythonized [@Cython] version of the algorithm which also allows for multi-process void-finding. The `Vsquared` package meanwhile uses the `scipy.spatial` [@SciPy] submodule for fast computation of the Voronoi tessellation and convex hulls involved in the algorithm.




# VoidFinder

`VoidFinder` is an implementation of the VoidFinder algorithm [@El-Ad:1997], which is based on the growth and union of empty spheres. This algorithm is motivated by the expectation that voids are spherical to first order. It begins by removing from the catalog all isolated galaxies, defined as having significantly ($1.5\sigma$) larger than average third-nearest neighbor distances. The remaining galaxies are then placed on a grid, and empty grid cells are identified. Next, empty spheres are grown starting from the centers of the empty cells. The resulting holes are sorted by radius, and the largest is identified as a maximal sphere -- the largest sphere that can fit in a given void region. Then each subsequent hole, if it overlaps no already-identified maximal sphere by more than 10\% and has a radius larger than $10h^{-1}\textnormal{Mpc}$, is also identified as a maximal sphere. Finally, holes not identified as maximal spheres are used to enhance the volume of each void.




# V<sup>2</sup>

`Vsquared` is a software package for finding voids based on the ZOBOV (ZOnes Bordering On Voidness) algorithm [@Neyrinck:2007], which grows voids from local minima in the density field. This algorithm first produces a Voronoi tessellation of the galaxy catalog, and the volumes of the Voronoi cells are used to identify local density minima. Zones are then grown from density minima in the distribution of cells using a watershed transform, where each cell is linked to its least dense neighbor. Finally, voids are formed from these by identifying low-density boundaries between adjacent zones and using them to grow unions of weakly divided zones. A void-pruning step is usually also run at the end of the algorithm, to remove void candidates unlikely to be true voids. A number of different implementations of this step exist, and `Vsquared' includes several.




# 3D Visualization

Both `VoidFinder` and `Vsquared` include OpenGL based visulization for their outputs. [in progress]
- Screen shot of VoidFinder and V2 voids in 3D using SDSS DR7 NGC. Maybe sub-select to show the largest voids only so it's not a hugely messy-looking image.
- Description of scriptable animation options available with ffmpeg3. Details avaialable in online documentation.
- Slice plots? Probably overkill to include a plot... just mention this is available?



# Acknowledgements

We acknowledge support from the DOE Office of High Energy Physics under grant number DE-SC0008475.




# References
