

############################################################
# Installation
############################################################
Requires OpenGL >= 1.2

'pip install vispy'

cd into VoidFinder/python/voidfinder/viz and run

'cython -a *.pyx'

'python setup.py build_ext --inplace'


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

mouse wheel - increase/decrease galaxy size

0 - (zero key) - save screen as png in current directory


############################################################
# Backlog:
############################################################


- video recording capability
- color by void group instead of all blue or constant
? add index maps from the vertex/triangle data buffers to which void hole they belong to so we can do fancy things
- embed in webpage


+ fix the png image save on press_0 function, manually add background color (white) in?
+ add some kind of visual shading or lighting or something to make it easier to see the void spheres as we move through the middle of the cluster
+ (Not necessary with new lighting model) color galaxies by void and wall
+ (Not necessary, can see inside voids now with new lighting model) Should do, to be able to see all interior void galaxies; Not going to do, existing way is efficient enough) update green color so that it just modifies the existing triangle colors instead of moving the green sphere around
+ (Not necessary with new lighting model - just set void alpha to opaque) make interior void galaxies opaque when camera is inside the void
+ add image save
+ color holes which camera is inside of with green so you know you're inside
    + (dont do this) set alpha to 1.0 for everything, then set it to .5, .4, .3 for holes that user is within ~ 50mpc of?
+ (had 0 hits) filter out holes which live completely within other holes (might fix problem where some sphere edges have holes in them for no apparent reason?)
+ reconfigure location within voidfinder repo
+ test on Mac again
+ Test on Mac - it works!  Just install vispy
+ Add control bar to canvas for increasing/decreasing mouse sensitivity and such
+ Add key-based mouse sensitivity and such
+ Add spheres instead of disks
+ Add ability to rotate camera instead of just the data






