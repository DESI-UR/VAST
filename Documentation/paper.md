---
title: 'VAST: the Void Analysis Software Toolkit'
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
    orcid: 0000-0003-1366-7325
    affiliation: 2
  - name: Segev BenZvi
    orcid: 0000-0001-5537-4710
    affiliation: 1
  - name: Fatima Zaidouni
    orcid: 0000-0003-0931-0868
    affiliation: 1
  - name: Michaela Guzzetti
    orcid: 0000-0001-6866-8291
    affiliation: "1","3","4"
affiliations:
  - name: University of Rochester
    index: 1
  - name: Independent Researcher
    index: 2
  - name: Smith College
    index: 3
  - name: University of Washington
    index: 4
date: September 2020
bibliography: paper.bib
---


# Summary

Voids are expansive regions in the universe containing significantly fewer galaxies than surrounding galaxy clusters and filaments. They are a fundamental feature of the cosmic web, and provide important information about galaxy physics and cosmology. For example, correlations between voids and luminous tracers of large-scale structure improve constraints on the expansion of the universe compared to using tracers alone. However, the basic concept of a void is vague and formulating a concrete definition to use in a void-finding algorithm is not trivial. As a result, several different algorithms exist to identify these cosmic underdensities. The Void Analysis Software Toolkit, or VAST, provides Python 3 implementations of two such algorithms in the `VoidFinder` and `Vsquared` packages.

Analyzing the next generation of cosmological surveys will require the ability to process large volumes of data (at least 25 million galaxies and quasars from the Dark Energy Spectroscopic Instrument [@DESI]). To more efficiently process such large datasets, the `VoidFinder` package includes a Cythonized [@Cython] version of the algorithm which also allows for multi-process void-finding. The `Vsquared` package meanwhile uses the `scipy.spatial` [@SciPy] submodule for fast computation of the Voronoi tessellation and convex hulls involved in the algorithm.




# VoidFinder

`VoidFinder` is an implementation of the VoidFinder algorithm [@El-Ad:1997], which is based on the growth and union of empty spheres. This algorithm is motivated by the expectation that voids are spherical to first order. It begins by removing from the catalog all isolated galaxies, defined as having significantly ($1.5\sigma$) larger than average third-nearest neighbor distances. The remaining galaxies are then placed on a grid, and empty grid cells are identified. Next, empty spheres are grown starting from the centers of the empty cells. The resulting holes are sorted by radius, and the largest is identified as a maximal sphere -- the largest sphere that can fit in a given void region. Then each subsequent hole, if it overlaps no already-identified maximal sphere by more than 10\% and has a radius larger than $10h^{-1}\textnormal{Mpc}$, is also identified as a maximal sphere. Finally, holes not identified as maximal spheres are used to enhance the volume of each void.




# V<sup>2</sup>

`Vsquared` is a software package for finding voids based on the ZOBOV (ZOnes Bordering On Voidness) algorithm [@Neyrinck:2007], which grows voids from local minima in the density field. This algorithm first produces a Voronoi tessellation of the galaxy catalog, and the volumes of the Voronoi cells are used to identify local density minima. Zones are then grown from density minima in the distribution of cells using a watershed transform, where each cell is linked to its least dense neighbor. Finally, voids are formed from these by identifying low-density boundaries between adjacent zones and using them to grow unions of weakly divided zones. A void-pruning step is usually also run at the end of the algorithm, to remove void candidates unlikely to be true voids. A number of different implementations of this step exist, and `Vsquared` includes several, including methods from other ZOBOV implementations such as VIDE [@VIDE:2012] and REVOLVER [@REVOLVER:2018].




# 3D Visualization

![VoidRender visualization of the output from SDSS DR7 [@Abazajian:2009].\label{fig:vfviz}](voidfinder_viz.png)

In order to aid in assessing the quality of the VoidFinder algorithm, the `voidfinder.viz` package includes a `VoidRender` class (`from voidfinder.viz import VoidRender`) which utilizes a combination of `OpenGL` and the python `vispy` package to enable real-time 3D rendering of the VoidFinder algorithm output. This 3D visualization allows the user to explore 3D space in a video-game-esque manner, where the w-a-s-d-style keyboard controls function to give the user a full range of motion: forward/backward/left/right/up/down translation and pitch/yaw/roll rotation. Each void hole of the VoidFinder output is rendered to the screen using the icosadehral sphere approximation, where the depth of the approximation is configurable and higher approximation depths yield a finer and finer grained triangularization of each sphere. In addition, the visualization includes an option to remove the interior walls of each void, which is approximated by removing the triangles from the triangluarization where all three vertices of a given triangle fall within the radius of an intersecting sphere. This option aids in visually inspecting the properties of joining spheres. The galaxy survey upon which the output Voids are based may also be included within the visualization, represented by small dots since the radius of even the largest galaxy is negligibly small compared to the radius of the smallest Void. For visual purposes, the mouse scroll wheel may be used to enlarge or shrink the galaxy dot size. By passing the appropriate portions of the galaxy survey to different parts of the VoidRender keyword parameters, wall galaxies may be displayed in black and void galaxies may be displayed in red. Additionally, in order to help visualize the clustering of wall galaxies, another VoidRender option will plot a thin black line between a galaxy and its K nearest neighbors, yielding a denser spider-web look for those galaxies which cluster together, as can be seen in Figure \autoref{fig:vfviz}.

![`Vsquared` visualization of the output from SDSS DR7.\label{fig:v2viz}](vsquared_viz.png)

An animated example of the VoidRender visualization can be found here: https://www.youtube.com/watch?v=PmyoUAt4Qa8. VoidRender can be utilized to produce screenshots or videos such as this example if a user's environment includes the `ffmpeg` library. `Vsquared` also includes an `OpenGL` and `vispy` based visualization for its output. The surfaces of voids found by the ZOBOV algorithm are made up of convex polygons, and are rendered exactly in 3D. Controls for movement and production of screenshots and videos are identical to those of VoidRender. An example of the `Vsquared` visualization is shown in Figure \autoref{fig:v2viz}.



# Acknowledgements

We acknowledge support from the DOE Office of High Energy Physics under grant number DE-SC0008475.




# References
