
"""
Visualization module for VoidFinder results.

Requires OpenGL >= 1.2
vispy >= 0.6.3

Usage:
------

from voidfinder.viz import VoidFinderCanvas, load_hole_data, load_galaxy_data

holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")

galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")

viz = VoidFinderCanvas(holes_xyz, 
                 holes_radii, 
                 galaxy_data,
                 canvas_size=(1600,1200))

viz.run()

"""
from .view_results import VoidFinderCanvas

from .load_results import load_hole_data, load_galaxy_data

