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
  - name: Dahlia Veyrat
    orcid: 0000-0001-8101-2836
    affiliation: 1
  - name: Stephen W. O'Neill Jr.
    orcid: 0000-0003-1366-7325
    affiliation: 2
  - name: Segev BenZvi
    orcid: 0000-0001-5537-4710
    affiliation: 1
  - name: Fatima Zaidouni
    orcid: 0000-0003-0931-0868
    affiliation: "1, 5"
  - name: Michaela Guzzetti
    orcid: 0000-0001-6866-8291
    affiliation: "1, 3, 4"
affiliations:
  - name: University of Rochester
    index: 1
  - name: Independent Researcher
    index: 2
  - name: Smith College
    index: 3
  - name: University of Washington
    index: 4
  - name: Massachusetts Institute of Technology
    index: 5
date: September 2022
bibliography: paper.bib
---


# Summary

Voids are expansive regions in the universe containing significantly fewer 
galaxies than surrounding galaxy clusters and filaments.  They are a fundamental 
feature of the cosmic web and provide important information about galaxy physics 
and cosmology.  For example, correlations between voids and luminous tracers of 
large-scale structure improve constraints on the expansion of the universe as 
compared to using tracers alone, and numerous studies have shown that the void 
environment influences the evolution of galaxies.  However, what constitutes a 
void is vague and formulating a concrete definition to use in a void-finding 
algorithm is not trivial.  As a result, several different algorithms exist to 
identify these cosmic underdensities.  Our Void Analysis Software Toolkit, or 
VAST, provides Python 3 implementations of two such algorithms: **VoidFinder** 
and **V<sup>2</sup>**.  This consolidation of two popular void-finding 
algorithms allows the user to, for example, easily compare the results of their 
analysis using different void definitions.


# Statement of Need

Analyzing the next generation of cosmological surveys will require the ability 
to process large volumes of data [for example, at least 30 million galaxies and 
quasars from the Dark Energy Spectroscopic Instrument, @DESI].  To more 
efficiently process such large datasets, this Python 3 implementation of 
**VoidFinder** includes a Cythonized [@Cython] version of the algorithm which 
also allows for multi-process void-finding.  When run on the 7th Data Release of 
the main galaxy sample of the Sloan Digital Sky Survey [SDSS DR7, 
@Abazajian:2009] on a single thread, `vast.voidfinder` requires less than 20 
seconds to run, compared to the ~3 hours needed to run the original Fortran 
version of **VoidFinder** (both run on an Intel Core i7-6700K @ 4GHz).  The 
void-finding algorithm in **V<sup>2</sup>** uses the `scipy.spatial` [@SciPy] 
submodule for fast computation of the Voronoi tessellation and convex hulls 
involved in the algorithm.  In addition, `vast.Vsquared` consolidates a large 
number of void-pruning methods (many currently available in separate programming 
packages and different languages) into a single package.




# VoidFinder

`vast.voidfinder` is a software package containing a Python 3 implementation of 
the Void Finder algorithm [@El-Ad:1997] that is based on the algorithm's Fortran 
implementation by @Hoyle:2002.  Motivated by the expectation that voids are 
spherical to first order, this algorithm defines voids as the unions of sets of 
spheres grown in the underdense regions of the large-scale structure.  It begins 
by removing all isolated tracers from the catalog of objects, defined as having 
significantly ($1.5\sigma$) larger than average third-nearest neighbor 
distances.  The remaining tracers are then placed on a grid, and a sphere is 
grown from the center of any empty grid cells until it is bounded by four 
tracers on its surface.  All spheres larger than a specified radius (typically 
around 10 Mpc/h) are considered possible maximal spheres -- the largest sphere 
that can fit in a given void region.  Filtering through these candidate maximal 
spheres by order of decreasing radius, no maximal sphere can overlap by more 
than 10% of its volume with any other previously identified (larger) maximal 
sphere.  After the maximal spheres are identified, all non-maximal spheres 
(holes) are combined with these maximal spheres to enhance the void structure if 
they overlap exactly one maximal sphere by at least 50% of its volume.  The 
union of a set of spheres (one maximal and the remaining smaller holes) defines 
a void region.




# V<sup>2</sup>

`vast.Vsquared` is a software package for finding voids based on the ZOBOV 
(ZOnes Bordering On Voidness) algorithm [@Neyrinck:2007], which grows voids from 
local minima in the density field.  This algorithm first produces a Voronoi 
tessellation of the catalog of large-scale tracers, and the volumes of the 
Voronoi cells are used to identify local density minima.  Zones are then built 
from density minima in the distribution of cells using a watershed transform, 
where each cell is linked to its least dense neighbor.  Finally, voids are 
formed from these by identifying low-density boundaries between adjacent zones 
and using them to grow unions of weakly divided zones.  This list of voids is 
then typically pruned to remove void candidates unlikely to be true voids.
`Vsquared` includes several of the different void-pruning methods that exist, 
including methods from other ZOBOV implementations such as VIDE [@VIDE:2012] and 
REVOLVER [@REVOLVER:2018].  The VIDE method, for example, sets a maximum density 
for the boundaries by which zones can be grown into voids.  Any voids formed 
from unions of zones with denser boundaries are removed from the catalog.




# VoidRender: a 3D Visualization of voids

In order to aid in assessing the quality of the **VoidFinder** algorithm, the 
`vast.voidfinder.viz` package includes a `VoidRender` class 
(`from vast.voidfinder.viz import VoidRender`) which utilizes a combination of 
`OpenGL` and the python `vispy` package [@vispy] to enable real-time 3D rendering 
of the **VoidFinder** algorithm output.  This 3D visualization allows the user to 
explore 3D space in a video-game-esque manner, where the w-a-s-d-style keyboard 
controls function to give the user a full range of motion: 
forward/backward/left/right/up/down translation and pitch/yaw/roll rotation.  
Each void hole of the **VoidFinder** output is rendered to the screen using the 
icosadehral sphere approximation, where the depth of the approximation is 
configurable and higher approximation depths yield a finer and finer grained 
triangularization of each sphere.  In addition, **VoidRender** includes an option 
to remove the interior walls of each void, which is approximated by removing the 
triangles from the triangluarization where all three vertices of a given 
triangle fall within the radius of an intersecting sphere.  This option aids in 
visually inspecting the properties of joining spheres.

The galaxy survey upon which the output voids are based may also be included 
within the visualization, where each galaxy is represented by a small dot, as 
the radius of even the largest galaxy is negligibly small compared to the radius 
of the smallest void.  For visual purposes, the mouse scroll wheel may be used 
to enlarge or shrink the galaxy dot size.  By passing the appropriate portions 
of the galaxy survey to different parts of the **VoidRender** keyword parameters, 
wall galaxies may be displayed in black and void galaxies may be displayed in 
red.  Additionally, in order to help visualize the clustering of wall galaxies, 
another **VoidRender** option plots a thin black line between a galaxy and its 
$k$ nearest neighbors, yielding a denser spider-web look for those galaxies which 
cluster together, as can be seen in \autoref{fig:vfviz}.

An animated example of the **VoidRender** visualization can be found on 
[YouTube](https://www.youtube.com/watch?v=PmyoUAt4Qa8).  **VoidRender** can be 
utilized to produce screenshots or videos such as this example if a user's 
environment includes the `ffmpeg` library.  **V<sup>2</sup>** also includes an 
`OpenGL`- and `vispy`-based visualization for its output.  The surfaces of voids 
found by the ZOBOV algorithm are made up of convex polygons, and are rendered 
exactly in 3D.  Controls for movement and production of screenshots and videos 
are identical to those of **VoidRender**.  An example of the **V<sup>2</sup>** 
visualization is shown in \autoref{fig:v2viz}.

![**VoidRender** visualization of the output from SDSS DR7 [@Abazajian:2009].  
Void regions are shown as the shaded colorful regions; each different color 
corresponds to a different void.  Black points are the non-isolated (wall) 
galaxies used to define the void regions, and red points show the field 
galaxies.  Any two wall galaxies that are closer than the maximum distance used 
to separate wall and field galaxies are connected by a black line.\label{fig:vfviz}](voidfinder_viz.png)

![**V<sup>2</sup>** visualization of the output from SDSS DR7.  Void regions are 
the large shaded polyhedra, and the galaxies are shown as red points.\label{fig:v2viz}](vsquared_viz.png)

# Acknowledgements

SB and DV acknowledge support from the DOE Office of High Energy Physics under 
award number DE-SC0008475.  MG was supported by the NSF Research Experience for 
Undergraduates program under award number 1757062.  FZ acknowledges support from 
the University of Rochester Research Innovation Grants program.


# References
