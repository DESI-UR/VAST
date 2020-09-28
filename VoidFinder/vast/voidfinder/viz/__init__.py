
"""
Visualization module for VoidFinder results.

Requires OpenGL >= 1.2
vispy >= 0.6.3
optionally requires ffmpeg available on PATH for video recording

Usage:
------

from voidfinder.viz import VoidRender, load_hole_data, load_galaxy_data

holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")

galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")

viz = VoidRender(holes_xyz, 
                 holes_radii, 
                 galaxy_data,
                 canvas_size=(1600,1200))

viz.run()

"""
from .void_render import VoidRender

from .load_results import load_hole_data, load_galaxy_data

