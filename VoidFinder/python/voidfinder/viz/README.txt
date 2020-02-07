

############################################################
# Installation
############################################################
Requires OpenGL >= 1.2

'pip install vispy'

cd into VoidFinder/python/voidfinder/viz and run

'cython -a *.pyx'

'python setup.py build_ext --inplace'

Optional (required for video recording):

Install ffmpeg

sudo apt install ffmpeg


############################################################
# Notes
############################################################
https://vispy-website.readthedocs.io/en/latest/
http://vispy.org/installation.html

On Ubuntu 18.04 Nvidia Drivers 430.50 CUDA 10.1

was getting segfault on "import vispy; vispy.sys_info()"

trying:

'sudo apt-get install mesa-utils'

didn't fix it, trying:

sudo apt-get install mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev

vispy.sys_info() worked!

Tutorial on OpenGL:
https://learnopengl.com/Getting-started/Hello-Triangle



############################################################
# Controls
############################################################


w - translate forward
s - translate backward
a - translate left
d - translate right
r - elevate up
f - elevate down

q - roll left
e - roll right
i - pitch up
k - pitch down
j - yaw left
k - yaw right

z - increase translation sensitivity
x - decrease translation sensitivity
c - increase rotation sensitivity
v - decrease rotation sensitivity

m - start/stop video recording NOTE:  MAY TAKE A LOT OF RAM
0 - screenshot

Left mouse click - pitch & yaw
Right mouse click - translate forward & backward
Mouse Wheel - increase & decrease galaxy size


############################################################
# Backlog:
############################################################



- Change typedefs.pxd imports in cython to from ..typedefs import 'whatever' since the code
    is part of the VoidFinder repo and VoidFinder cython already has a typedefs.pxd module
    which is exactly the same
- Investigate why the new sorted-max-sphere based neighborization approach to
    sphere vertex intersection removal is resulting in slightly more triangles remaining
    after the removal (7.9 million versus 7.8 million)
- Embed the VoidRender canvas inside a larger PyQt program maybe so we can add slider bars
    for movement sensitvity and mouse-clickable buttons along a bottom control panel, and
    maybe a display showing position or orientation or something?
- Investigate max number of plottable vertices, maybe an OpenGL setting or hardware limit
      VoidRender was dropping spheres on laptop with remove_void_intersect turned off
- Finish the "Seam Adjustment" code so that it actually works
- Adjust array of normal vectors so that it doesnt contain 3 copies of each normal,
    instead just keep 1 copy and add an indexbuffer that points all 3 vertices to
    the same row?  I think this is possible in openGL just need to review that
    process and see if I can get it working, will save memory.
- Adjust vertex buffers for position and normal and such so that we can remove the 4th
    column of unused data to save memory, just add a vec4(pos, 1.0) in the shaders
- embed in webpage
- (maybe do this, maybe not?) add index maps from the vertex/triangle data buffers to which 
    void hole they belong to so we can do fancy things



+ Added reference/orientation sphere to rendering
+ Fix alpha compositing in screenshot/video
   (Investigated, think its actually being done correctly because all alpha values were 255
   when using the self.read_front_buffer() method)
+ Attempt to interleave the Nearest Neighbor queries in order starting from maximum to minimum sized objects so that
    as we iterate through, we can reduce the size of the next radius_query() since the new maximum radius will be
    smaller than the original maximum radius since that will have been exhausted by doing the max sphere first
    (Completed this and while it reduced number of neighbors to search from like 13e6 down to like 2.5e6 it only saved
    a few seconds, from like 16 seconds down to 14 seconds on 27,125 holes sphere_depth==3 since I guess the vast
    majority of the time in the intersection removal code is spent elsewhere.  Minor improvement though).
+ Add option to have intersect removal code only remove intersections within same Void Group
+ video recording capability
+ (Done outside of VoidRender class) color by void group instead of all blue or constant
+ added mouse navigation (left click pitch/yaw right click forward/back)
+ fix the png image save on press_0 function, manually add background color (white) in?
+ add some kind of visual shading or lighting or something to make it easier to see the void spheres as we move 
    through the middle of the cluster
+ (Not necessary with new lighting model) color galaxies by void and wall
+ (Not necessary, can see inside voids now with new lighting model) Should do, to be able to see all interior void 
    galaxies; Not going to do, existing way is efficient enough) update green color so that it just modifies the 
    existing triangle colors instead of moving the green sphere around
+ (Not necessary with new lighting model - just set void alpha to opaque) make interior void galaxies opaque when 
    camera is inside the void
+ add image save
+ color holes which camera is inside of with green so you know you're inside
    + (dont do this) set alpha to 1.0 for everything, then set it to .5, .4, .3 for holes that user is within ~ 50mpc of?
+ (had 0 hits) filter out holes which live completely within other holes (might fix problem where some sphere edges 
    have holes in them for no apparent reason?)
+ reconfigure location within voidfinder repo
+ test on Mac again
+ Test on Mac - it works!  Just install vispy
+ Add control bar to canvas for increasing/decreasing mouse sensitivity and such
+ Add key-based mouse sensitivity and such
+ Add spheres instead of disks
+ Add ability to rotate camera instead of just the data






