

import os

import subprocess

import numpy as np

from load_results import load_hole_data, load_galaxy_data

from unionize import union_vertex_selection

from vispy import gloo

from vispy import app

import vispy.io as io

from vispy.util.transforms import perspective, translate, rotate

#from vispy.util.quaternion import Quaternion

#from vispy.visuals.transforms import STTransform

from vispy.color import Color

#from vispy.geometry import create_box, create_sphere

#from vispy import scene

from sklearn import neighbors

import time

vert = """
#version 120
// Uniforms
// ------------------------------------
//uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_linewidth;
uniform float u_antialias;
uniform float u_size;
// Attributes
// ------------------------------------
//attribute vec3  a_position;
attribute vec4  a_position;
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;
attribute float a_size;
// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;
void main (void) {
    v_size = a_size * u_size;
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;
    v_fg_color  = a_fg_color;
    v_bg_color  = a_bg_color;
    
    //gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    //gl_Position = u_projection * u_view * u_model * a_position;
    gl_Position = u_projection * u_view * a_position;
    
    gl_PointSize = v_size + 2.*(v_linewidth + 1.5*v_antialias);
}
"""

frag = """
#version 120
// Constants
// ------------------------------------
// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;
// Functions
// ------------------------------------
// ----------------
float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2.;
    return r;
}
// ----------------
float arrow_right(vec2 P, float size)
{
    float r1 = abs(P.x -.50)*size + abs(P.y -.5)*size - v_size/2.;
    float r2 = abs(P.x -.25)*size + abs(P.y -.5)*size - v_size/2.;
    float r = max(r1,-r2);
    return r;
}
// ----------------
float ring(vec2 P, float size)
{
    float r1 = length((P.xy - vec2(0.5,0.5))*size) - v_size/2.;
    float r2 = length((P.xy - vec2(0.5,0.5))*size) - v_size/4.;
    float r = max(r1,-r2);
    return r;
}
// ----------------
float clober(vec2 P, float size)
{
    const float PI = 3.14159265358979323846264;
    const float t1 = -PI/2.;
    const vec2  c1 = 0.2*vec2(cos(t1),sin(t1));
    const float t2 = t1+2.*PI/3.;
    const vec2  c2 = 0.2*vec2(cos(t2),sin(t2));
    const float t3 = t2+2.*PI/3.;
    const vec2  c3 = 0.2*vec2(cos(t3),sin(t3));
    float r1 = length((P.xy- vec2(0.5,0.5) - c1)*size);
    r1 -= v_size/3;
    float r2 = length((P.xy- vec2(0.5,0.5) - c2)*size);
    r2 -= v_size/3;
    float r3 = length((P.xy- vec2(0.5,0.5) - c3)*size);
    r3 -= v_size/3;
    float r = min(min(r1,r2),r3);
    return r;
}
// ----------------
float square(vec2 P, float size)
{
    float r = max(abs(P.x -.5)*size,
                  abs(P.y -.5)*size);
    r -= v_size/2.;
    return r;
}
// ----------------
float diamond(vec2 P, float size)
{
    float r = abs(P.x -.5)*size + abs(P.y -.5)*size;
    r -= v_size/2.;
    return r;
}
// ----------------
float vbar(vec2 P, float size)
{
    float r1 = max(abs(P.x -.75)*size,
                   abs(P.x -.25)*size);
    float r3 = max(abs(P.x -.5)*size,
                   abs(P.y -.5)*size);
    float r = max(r1,r3);
    r -= v_size/2.;
    return r;
}
// ----------------
float hbar(vec2 P, float size)
{
    float r2 = max(abs(P.y -.75)*size,
                   abs(P.y -.25)*size);
    float r3 = max(abs(P.x -.5)*size,
                   abs(P.y -.5)*size);
    float r = max(r2,r3);
    r -= v_size/2.;
    return r;
}
// ----------------
float cross(vec2 P, float size)
{
    float r1 = max(abs(P.x -.75)*size,
                   abs(P.x -.25)*size);
    float r2 = max(abs(P.y -.75)*size,
                   abs(P.y -.25)*size);
    float r3 = max(abs(P.x -.5)*size,
                   abs(P.y -.5)*size);
    float r = max(min(r1,r2),r3);
    r -= v_size/2.;
    return r;
}
// Main
// ------------------------------------
void main()
{
    float size = v_size +2.0*(v_linewidth + 1.5*v_antialias);
    
    float t = v_linewidth/2.0-v_antialias;
    
    float r = disc(gl_PointCoord, size);
    // float r = square(gl_PointCoord, size);
    // float r = ring(gl_PointCoord, size);
    // float r = arrow_right(gl_PointCoord, size);
    // float r = diamond(gl_PointCoord, size);
    // float r = cross(gl_PointCoord, size);
    // float r = clober(gl_PointCoord, size);
    // float r = hbar(gl_PointCoord, size);
    // float r = vbar(gl_PointCoord, size);
    
    float d = abs(r) - t;
    
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0.)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""



vert_sphere = """

#version 120

uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec4 position;
attribute vec4 normal;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    //remember - column major indexing in GLSL
    //since its column major, the u_view matrix is essentially the transpose
    //of the self.view matrix in the python below.  So to get the xyz positions we're
    //going to pull them from the 4th column instead of the 4th row, and when we do the
    //matrix multiplication to convert from view space to world space position, to
    //multiply on the left with the u_view matrix we need to use the transpose:
    //vec4 curr_camera_position = -1.0 * transpose(u_view) * view_camera_position;
    //or we can multiply on the right with it natively
    
    vec4 view_camera_position = vec4(u_view[3][0], u_view[3][1], u_view[3][2], 0.0f);
    
    vec4 curr_camera_position = -1.0 * view_camera_position * u_view;
    
    vec4 position_difference = curr_camera_position - position;
    
    position_difference[3] = 0.0f;
    
    float position_distance = length(position_difference);
    
    float angle_color_mod = abs(dot(normalize(position_difference), normal));
    
    //float dist_color_mod = 100.0/(position_distance*position_distance);
    //dist_color_mod = min(1.0, dist_color_mod);
    
    
    float dist_color_mod = 1.0f;
    
    if(position_distance > 2.719)
    {
        float dist_color_mod = 1.0/log(2.0*position_distance);
    }
    
    
    
    angle_color_mod = angle_color_mod * 0.8 + 0.2;
    
    dist_color_mod = dist_color_mod * 0.9 + 0.1;
    
    float color_mod = angle_color_mod*dist_color_mod;
    
    v_color.xyz = color_mod*color.xyz;
    v_color.w = color.w;
    
    gl_Position = u_projection * u_view * position;
}
"""

frag_sphere = """

#version 120

varying vec4 v_color;

void main()
{

    gl_FragColor = v_color;
    //gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

"""


class Triangle(object):
    
    def __init__(self, pt1, pt2, pt3):
        
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3








# ------------------------------------------------------------ Canvas class ---
class VoidRender(app.Canvas):

    def __init__(self,
                 holes_xyz, 
                 holes_radii, 
                 galaxy_xyz,
                 galaxy_display_radius=2.0,
                 remove_void_intersects=True,
                 filter_for_degenerate=True,
                 canvas_size=(800,600),
                 title="VoidFinder Results",
                 camera_start_location=None,
                 camera_start_orientation=None,
                 start_translation_sensitivity=1.0,
                 start_rotation_sensitivity=1.0,
                 galaxy_color=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                 void_hole_color=np.array([0.0, 0.0, 1.0, 0.95], dtype=np.float32),
                 #enable_void_interior_highlight=True,
                 #void_highlight_color=np.array([0.0, 1.0, 0.0, 0.3], dtype=np.float32),
                 SPHERE_TRIANGULARIZATION_DEPTH=3
                 ):
        '''
        Main class for initializing the visualization.
        
        Usage:
        ------
        
        from voidfinder.viz import VoidRender, load_hole_data, load_galaxy_data
        
        holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")
    
        galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
    
        viz = VoidFinderCanvas(holes_xyz, 
                         holes_radii, 
                         galaxy_data,
                         canvas_size=(1600,1200))
    
        viz.run()
        
        Notes
        -----
        
        Controls:
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
        
        m - start/stop video recording NOTE:  MAY TAKE A LOT OF RAM
        0 - screenshot
        
        Left mouse click - pitch & yaw
        Right mouse click - translate forward & backward
        Mouse Wheel - increase & decrease galaxy size
        
        Parameters
        ----------
        
        holes_xyz : (N,3) numpy.ndarray
            xyz coordinates of the hole centers
            
        holes_radii : (N,) numpy.ndarray
            length of the hole radii in xyz coordinates
            
        galaxy_xyz : (N,3) numpy.ndarray
            xyz coordinates of the galaxy locations
            
        galaxy_display_radius : float
            using a constant radius to display galaxy points since they should
            all be small compared to the void holes, and they don't have
            corresponding radii
            
        remove_void_intersects : bool, default True
            turn on (True) or off (False) the clipping of display triangles for
            void interiors.  When true, removes all the triangles of a void hole
            sphere which are fully contained inside another void hole, which essentially
            creates a surface union of void holes which overlap.  Note that for smaller
            sphere density values, the edge artifacts along the seams where void holes
            intersect will be more visually apparent
            
        filter_for_degenerate : bool, default True
            if true, filter the holes_xyz and holes_radii for any holes which
            are completely contained within another hole and remove them
            
        enable_void_interior_highlight : bool, default True
            if True, when a user enters a void sphere, it will highlight the sphere
            in a different color (default green) so a user knows they are looking out
            from inside a sphere.  False disables this functionality
            
        canvas_size : 2-tuple
            (width, height) in pixels for the output visualization
            
        title : str
            value to display at top of the window
            
        camera_start_location : ndarray of shape (3,) dtype float32
            starting location for the camera.  Remember to multiply by -1.0 to go
            from data coordinates to camera coordinates
            example: np.zeros(3, dtype=np.float32)
            
        camera_start_orientation : ndarray of shape (3,3) dtype float32
            rotation matrix describing initial camera orientation
            example: np.eye(3, dtype=np.float32)
            
        start_translation_sensitivity : float
            starting sensitivity for translation keys - wasdrf
            
        start_rotation_sensitivity : float
            starting sensitivity for rotation keys - qeijkl
            
        galaxy_color : ndarray shape (4,) dtype float32
           values from 0.0-1.0 representing RGB and Alpha for the galaxy
           points
           
        void_hole_color : ndarray shape (4,) or (num_void,4) dtype float32
           values from 0.0-1.0 representing RGB and Alpha for the voids
           
        void_highlight_color : ndarray shape (4,) dtype float32
           values from 0.0-1.0 representing RGB and Alpha for the void highlight,
           when the camera enters a void hole it turns the hole this color so you
           know you're inside IF enable_void_interior_highlight == True
           
        SPHERE_TRIANGULARIZATION_DEPTH : integer, default 3
           number of subdivisions in the icosahedron sphere triangularization to make.
           Total vertices will be num_voids * 20 * 3 * 4^SPHERE_TRIANGULARIZATION_DEPTH
           so be careful increasing this value above 3.  Default of 3 results in 3840 vertices
           (1280 triangles) per sphere
        '''
        
        
        app.Canvas.__init__(self, 
                            keys='interactive', 
                            size=canvas_size)
        
        self.title = title
        
        self.translation_sensitivity = start_translation_sensitivity
        
        self.rotation_sensitivity = start_rotation_sensitivity
        
        self.max_galaxy_display_radius = galaxy_display_radius
        
        self.holes_xyz = holes_xyz
        
        self.holes_radii = holes_radii
        
        if filter_for_degenerate:
            
            self.filter_degenerate_holes()
        
        self.galaxy_xyz = galaxy_xyz
        
        self.remove_void_intersects = remove_void_intersects
        
        #self.enable_void_interior_highlight = enable_void_interior_highlight
        
        self.num_hole = holes_xyz.shape[0]
        
        self.num_gal = galaxy_xyz.shape[0]
        
        self.galaxy_color = galaxy_color
        
        self.void_hole_color = void_hole_color
        
        #self.void_highlight_color = void_highlight_color
        
        #self.void_highlight_alpha = void_highlight_color[3]
        
        ######################################################################
        #
        ######################################################################
        
        self.projection = np.eye(4, dtype=np.float32) #start with orthographic, will modify this
        
        self.unit_sphere, self.unit_sphere_normals = self.create_sphere(1.0, SPHERE_TRIANGULARIZATION_DEPTH)
        
        self.vert_per_sphere = self.unit_sphere.shape[0]
        
        self.hole_kdtree = neighbors.KDTree(self.holes_xyz)
        
        
        
        ######################################################################
        #
        ######################################################################
        self.setup_camera_view(camera_start_location, camera_start_orientation)
        
        self.setup_mouse()
        
        self.setup_keyboard()
        
        self.recording = False
        
        self.recording_frames = []
        
        ######################################################################
        #
        ######################################################################
        
        self.setup_galaxy_program()
        
        self.setup_void_sphere_program()
        
        #if self.enable_void_interior_highlight:
            
        #    self.setup_highlight_program()
        
        self.apply_zoom()
        
        gloo.set_state('translucent', clear_color='white')
        
        ######################################################################
        # Set up a callback to a timer function, mostly to work on keyboard
        # input at this point
        # Then show the canvas
        ######################################################################
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        
        self.show()
        
    def setup_keyboard(self):
        
        self.last_keypress_time = time.time()
        
        
        
        self.keyboard_commands = {"w" : self.press_w,
                                  "s" : self.press_s,
                                  "a" : self.press_a,
                                  "d" : self.press_d,
                                  'r' : self.press_r,
                                  'f' : self.press_f,
                                  "z" : self.press_z,
                                  "x" : self.press_x,
                                  "c" : self.press_c,
                                  "v" : self.press_v,
                                  'i' : self.press_i,
                                  'k' : self.press_k,
                                  'l' : self.press_l,
                                  'j' : self.press_j,
                                  'q' : self.press_q,
                                  'e' : self.press_e}
        
        self.press_once_commands = {" " : self.press_spacebar,
                                    "p" : self.press_p,
                                    "m" : self.press_m,
                                    "0" : self.press_0}
        
        self.keyboard_active = {"w" : 0,
                                "s" : 0,
                                "a" : 0,
                                "d" : 0,
                                "r" : 0,
                                "f" : 0,
                                "z" : 0,
                                "x" : 0,
                                "c" : 0,
                                "v" : 0,
                                "i" : 0,
                                "k" : 0,
                                "l" : 0,
                                "j" : 0,
                                "q" : 0,
                                "e" : 0,
                                }
        
        
    def setup_mouse(self):
        
        
        self.point_size_denominator = 1.0
        
        self.mouse_state = 0
        
        self.last_mouse_pos = [0.0, 0.0]




    def setup_camera_view(self, start_location=None, start_orientation=None):
        """
        Set up the initial camera location.  Camera is controlled by setting
        the uniform variable 'u_view' on the program objects, example:
        
        'self.program['u_view'] = self.view' 
        
        which become part of the final object positions:
        
        old: "gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);"
        new: "gl_Position = u_projection * u_view * a_position;"
        
        Note - removed the 'u_model' matrix since it added too much complexity what with
               all the new functionality I've added to the camera (fly around, interior
               void green highlight, etc) and made a_position natively a 4-vector
        
        
        Directions: (x,y,z)
           z axis - forward/backward
           y axis - up/down
           x axis - left/right
    
        All camera information is encoded in the 4x4 matrix 'self.view'.
        
        self.view[0:3, 0:3] - encodes a 3x3 rotation matrix
        
        self.view[3, 0:3] - encodes the 1x3 (x,y,z) coordinate of the camera in the 
                            rotation matrix's frame of reference.
                            
        self.view[0:3, 3] - might just be always [0,0,0]?
        
        Also need to multiply by -1.0 to go from camera frame of reference to data 
        coordinates (since "moving the camera" is really "moving everything in the 
        scene in the opposite direction")
        
        ---------------------------------
        Example decoding actual location:
        ---------------------------------
        
        curr_rot_matrix = self.view[0:3,0:3]
        
        view_camera_location = self.view[3,0:3].reshape(1,3)
        
        curr_camera_location = -1.0*np.matmul(curr_rot_matrix, view_camera_location.T).T
        
        x = curr_camera_location[0]
        y = curr_camera_location[1]
        z = curr_camera_location[2]
        
        """
        
        if start_location is None:
        
            start_location = np.mean(self.holes_xyz, axis=0)
        
            start_location[2] += 300.0
            
            start_location *= -1.0
        
        
        self.view = np.eye(4, dtype=np.float32)
        
        self.view[3,0:3] = start_location
        
        
        if start_orientation is not None:
            
            self.view[0:3,0:3] = start_orientation
        
        
        
        

    def setup_galaxy_program(self):
        """
        Set up the OpenGL program via vispy that will display each of the
        galaxy coordinates provided in xyz-space as a gl_PointCoord
        
        # Set up some numpy arrays to work with the vispy/OpenGL vertex 
        # shaders and fragment shaders.  The names (string variables) used
        # in the below code match up to names used in the vertex and
        # fragment shader code strings used above
        
        # Create the color arrays in RGBA format for the holes and galaxies
        # I believe bg color is 'background' and fg is 'foreground' color
        
        # Since the data is currently being displayed as disks instead of
        # Spheres, use a fixed radius of 2.0 for galaxies for now since they
        # are small compared to void holes
        #
        """
        
        
        
        ######################################################################
        # Set up the vertex buffer backend
        ######################################################################
        self.galaxy_vertex_data = np.zeros(self.num_gal, [('a_position', np.float32, 4),
                                                          ('a_bg_color', np.float32, 4),
                                                          ('a_fg_color', np.float32, 4),
                                                          ('a_size', np.float32)])
        
        w_col = np.ones((self.num_gal, 1), dtype=np.float32)
        
        self.galaxy_vertex_data['a_position'] = np.concatenate((self.galaxy_xyz, w_col), axis=1)
        
        self.galaxy_vertex_data['a_bg_color'] = np.tile(self.galaxy_color, (self.num_gal,1))
        
        black = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) #black color
        
        self.galaxy_vertex_data['a_fg_color'] = np.tile(black, (self.num_gal,1)) 
        
        self.galaxy_vertex_data['a_size'] = self.max_galaxy_display_radius #broadcast to whole array?
        
        self.galaxy_point_VB = gloo.VertexBuffer(self.galaxy_vertex_data)
        
        ######################################################################
        # Set up the program to display galaxy points
        ######################################################################
        u_linewidth = 0.0
        
        u_antialias = 0.0 #0.0==turned this off it looks weird
        
        self.galaxy_point_program = gloo.Program(vert, frag)
    
        self.galaxy_point_program.bind(self.galaxy_point_VB)
        
        self.galaxy_point_program['u_linewidth'] = u_linewidth
        
        self.galaxy_point_program['u_antialias'] = u_antialias
        
        self.galaxy_point_program['u_view'] = self.view
        
        self.galaxy_point_program['u_size'] = 1.0

        
        
    def setup_void_sphere_program(self):
        """
        This vispy program draws all the triangles which compose the spheres
        representing the void holes
        """
        
        
        ######################################################################
        # Initialize some space to hold all the vertices (and w coordinate)
        # for all the vertices of all the spheres for all the void holes
        ######################################################################
        num_sphere_verts = self.vert_per_sphere*self.holes_xyz.shape[0]
        
        #num_sphere_triangles = num_sphere_verts//3
        
        self.void_sphere_coord_data = np.ones((num_sphere_verts,4), dtype=np.float32)
        
        self.void_sphere_coord_map = np.empty(num_sphere_verts, dtype=np.int64)
        
        self.void_sphere_normals_data = np.zeros((num_sphere_verts,4), dtype=np.float32)
        
        ######################################################################
        # Calculate all the sphere vertex positions and add them to the
        # vertex array, and the centroid of each triangle
        # ERRRT - don't need centroids, they're close enough to the vertices
        # themselves, just use the vertex positions, and a 
        # a copy of the normals!
        ######################################################################
        for idx, (hole_xyz, hole_radius) in enumerate(zip(self.holes_xyz, self.holes_radii)):
            
            curr_sphere = (self.unit_sphere * hole_radius) + hole_xyz
            
            start_idx = idx*self.vert_per_sphere
            
            end_idx = (idx+1)*self.vert_per_sphere
            
            self.void_sphere_coord_data[start_idx:end_idx, 0:3] = curr_sphere
            
            self.void_sphere_coord_map[start_idx:end_idx] = idx
            
            self.void_sphere_normals_data[start_idx:end_idx, 0:3] = self.unit_sphere_normals
            
            #for jdx in range(num_sphere_triangles):
                
                #self.void_sphere_centroid_data[kdx] = np.mean(curr_sphere[jdx:(jdx+3)], axis=0)
                
            
        ######################################################################
        # Given there will be a lot of overlap internal to the spheres, 
        # remove the overlap for better viewing quality
        ######################################################################
        if self.remove_void_intersects:
            
            print("Pre intersect-remove vertices: ", self.void_sphere_coord_data.shape[0])
            
            start_time = time.time()
            
            self.remove_hole_intersect_data()
            
            remove_time = time.time() - start_time
            
            num_sphere_verts = self.void_sphere_coord_data.shape[0]
            
            print("Post intersect-remove vertices: ", num_sphere_verts, "time: ", remove_time)
        
        ######################################################################
        #
        ######################################################################
        
        self.void_sphere_vertex_data = np.zeros(num_sphere_verts, [('position', np.float32, 4),
                                                                   ('normal', np.float32, 4),
                                                                   ('color', np.float32, 4)])
        
        self.void_sphere_vertex_data["position"] = self.void_sphere_coord_data
        
        self.void_sphere_vertex_data["normal"] = self.void_sphere_normals_data
        
        if self.void_hole_color.shape[0] == self.num_hole:
            
            void_hole_colors = np.empty((self.void_sphere_coord_data.shape[0], 4), dtype=np.float32)
            
            for idx, hole_idx in enumerate(self.void_sphere_coord_map):
                
                void_hole_colors[idx,:] = self.void_hole_color[hole_idx,:]
            
        else:
        
            void_hole_colors = np.tile(self.void_hole_color, (self.void_sphere_coord_data.shape[0], 1))
        
        self.void_sphere_vertex_data["color"] = void_hole_colors
        
        self.void_sphere_VB = gloo.VertexBuffer(self.void_sphere_vertex_data)
        
        ######################################################################
        # Set up the sphere-drawing program
        ######################################################################
        
        self.void_sphere_program = gloo.Program(vert_sphere, frag_sphere)
        
        self.void_sphere_program.bind(self.void_sphere_VB)
        
        self.void_sphere_program['u_view'] = self.view
        
        
        
        
        
        

        
    def setup_highlight_program(self):
        """
        Setup the sphere for when a user enters a void so it colors it
        green (or a different user-specified color) so you know you're 
        inside a void
        """
        
        self.highlight_state = -1
        
        self.highlight_sphere_coord_data = np.ones((self.unit_sphere.shape[0],4), dtype=np.float32)
        
        self.highlight_sphere_coord_data[:,0:3] = 10.0*self.unit_sphere
        
        ######################################################################
        #
        ######################################################################
        self.highlight_sphere_vertex_data = np.zeros(self.unit_sphere.shape[0], [('position', np.float32, 4),
                                                                                 ('color', np.float32, 4)])
        
        self.highlight_sphere_vertex_data["position"] = self.highlight_sphere_coord_data
        
        self.highlight_sphere_vertex_data["color"] = np.tile(self.void_highlight_color, (self.highlight_sphere_coord_data.shape[0], 1))
        
        self.highlight_sphere_vertex_data["color"][:,3] = 0.0
        
        self.highlight_sphere_VB = gloo.VertexBuffer(self.highlight_sphere_vertex_data)
        
        ######################################################################
        #
        ######################################################################
        self.highlight_program = gloo.Program(vert_sphere, frag_sphere)

        self.highlight_program.bind(self.highlight_sphere_VB)
        
        self.highlight_program['u_view'] = self.view
        
        
        
        
    def create_sphere(self, radius, subdivisions=2):
        """
        Could put other methods in here, since this is now basically just a wrapper
        function
        """
        
        sphere_vertices, sphere_normals = self.icosahedron_sphere_projection(radius, subdivisions)
        
        return sphere_vertices, sphere_normals
    
    
        
    def icosahedron_sphere_projection(self, radius, subdivisions):
        """
        Fixed the method from vispy so that it doesn't contain any interior triangles,
        just the exterior shell.
        
        Starting from an icosahedron of arbitrary size, project all the vertices radially onto
        the unit sphere.  Then, subdivide each face triangle by creating 4 sub-triangles as
        follows:
        
            /\               /\
           /  \             /__\
          /    \   -->     /\  /\
         /______\         /__\/__\  
        
        Tried the centerpoint 3-triangle decomposition but that results in really long spindly
        surfaces which just aren't good looking, the 4-triangle division looks much 
        smoother and better overall.
        
        Once each triangle has been subdivided, project the 3 new midpoints onto the
        unit sphere, discard the parent triangle and add the 4 new triangles to the
        face list.
        
        Number of output vertices = 20*3*(4^subdivisions)
        
        """
        # golden ratio
        t = (1.0 + np.sqrt(5.0))/2.0
    
        # vertices of an icosahedron
        verts = [(-1, t, 0),
                 (1, t, 0),
                 (-1, -t, 0),
                 (1, -t, 0),
                 (0, -1, t),
                 (0, 1, t),
                 (0, -1, -t),
                 (0, 1, -t),
                 (t, 0, -1),
                 (t, 0, 1),
                 (-t, 0, -1),
                 (-t, 0, 1)]
    
        # index into the vertices list above to get the
        # faces of the icosahedron
        face_idxs = [(0, 11, 5),
                 (0, 5, 1),
                 (0, 1, 7),
                 (0, 7, 10),
                 (0, 10, 11),
                 (1, 5, 9),
                 (5, 11, 4),
                 (11, 10, 2),
                 (10, 7, 6),
                 (7, 1, 8),
                 (3, 9, 4),
                 (3, 4, 2),
                 (3, 2, 6),
                 (3, 6, 8),
                 (3, 8, 9),
                 (4, 9, 5),
                 (2, 4, 11),
                 (6, 2, 10),
                 (8, 6, 7),
                 (9, 8, 1)]
    
        ############################################################
        # Put the initial icosahedron vertices on the unit sphere
        ############################################################
        unit_verts = np.zeros((12,3), dtype=np.float32)
        
        for idx, vert in enumerate(verts):
            
            modulus = np.sqrt(vert[0]*vert[0] + vert[1]*vert[1] + vert[2]*vert[2])
            
            unit_verts[idx, 0] = vert[0]/modulus
            unit_verts[idx, 1] = vert[1]/modulus
            unit_verts[idx, 2] = vert[2]/modulus
        
        ############################################################
        # Build a list of faces
        ############################################################
    
        face_list = []
        
        for face_idx in face_idxs:
            
            pt1 = unit_verts[face_idx[0]]
            pt2 = unit_verts[face_idx[1]]
            pt3 = unit_verts[face_idx[2]]
            
            face_list.append(Triangle(pt1, pt2, pt3))
            
        ############################################################
        # Subdivide each triangle - method 1 (centerpoint, 3-triangle)
        # makes some long thin triangles that dont look that great
        ############################################################
        '''
        for _ in range(subdivisions):
    
            temp_list = []
            
            for curr_triangle in face_list:
                
                v1 = curr_triangle.pt1
                v2 = curr_triangle.pt2
                v3 = curr_triangle.pt3
                
                midpoint = v1 + v2 + v3
                
                u_midpoint = midpoint/np.sqrt(np.sum(midpoint*midpoint))
                
                
                tri1 = Triangle(v1, v2, u_midpoint)
                tri2 = Triangle(v2, v3, u_midpoint)
                tri3 = Triangle(v3, v1, u_midpoint)
                
                temp_list.append(tri1)
                temp_list.append(tri2)
                temp_list.append(tri3)
                
            face_list = temp_list
        '''
        ############################################################
        # Subdivide each triangle - method 2 (4-triangle)
        ############################################################
        for _ in range(subdivisions):
    
            temp_list = []
            
            for curr_triangle in face_list:
                
                v1 = curr_triangle.pt1
                v2 = curr_triangle.pt2
                v3 = curr_triangle.pt3
                
                
                m1 = v1 + v2
                u_m1 = m1/np.sqrt(np.sum(m1*m1))
                
                m2 = v2 + v3
                u_m2 = m2/np.sqrt(np.sum(m2*m2))
                
                m3 = v3 + v1
                u_m3 = m3/np.sqrt(np.sum(m3*m3))
                
                
                
                
                tri1 = Triangle(v1, u_m1, u_m3)
                tri2 = Triangle(v2, u_m1, u_m2)
                tri3 = Triangle(v3, u_m2, u_m3)
                tri4 = Triangle(u_m1, u_m2, u_m3)
                
                temp_list.append(tri1)
                temp_list.append(tri2)
                temp_list.append(tri3)
                temp_list.append(tri4)
                
            face_list = temp_list
            
        ############################################################
        # Convert from a list of triangles to a numpy array
        # And calculate the normal vector for each triangle.  Copy
        # the normal vector for each triangle for each of the 3
        # vertices
        ############################################################
        
        array_data = []
        
        normal_vectors = []
        
        for curr_tri in face_list:
            
            array_data.append(curr_tri.pt1.reshape(1,3))
            array_data.append(curr_tri.pt2.reshape(1,3))
            array_data.append(curr_tri.pt3.reshape(1,3))
            
            
            edge_1 = curr_tri.pt2 - curr_tri.pt1
            edge_2 = curr_tri.pt3 - curr_tri.pt1
            
            normal_vec = np.cross(edge_1, edge_2)
            
            normal_vec /= np.sqrt(np.sum(normal_vec*normal_vec))
            
            #Copy once for each vertex
            normal_vectors.append(normal_vec.reshape(1,3))
            normal_vectors.append(normal_vec.reshape(1,3))
            normal_vectors.append(normal_vec.reshape(1,3))
            
        out_vertices = np.concatenate(array_data, axis=0)
        
        out_normals = np.concatenate(normal_vectors, axis=0)
        
        ############################################################
        # Scale by radius
        ############################################################
        
        out_vertices *= radius
        
        return out_vertices, out_normals

    
    
    
    
    def filter_degenerate_holes(self):
        
        hole_kdtree = neighbors.KDTree(self.holes_xyz)
        
        valid_idx = np.ones(self.holes_xyz.shape[0], dtype=np.bool)
        
        for curr_idx, (hole_xyz, hole_radius) in enumerate(zip(self.holes_xyz, self.holes_radii)):
            
            close_index = hole_kdtree.query_radius(hole_xyz.reshape(1,3), hole_radius)
            
            close_index = close_index[0]
            #[52, 128, 33, 1007, 4556]
            
            valid_close_index = close_index[close_index != curr_idx]
            #[52, 128, 33, 1007, 4556]
            
            neighbor_locations = self.holes_xyz[valid_close_index]
            
            neighbor_radii = self.holes_radii[valid_close_index]
            
            component_distances = neighbor_locations - hole_xyz
            
            
            neighbor_distances = np.sqrt(np.sum(component_distances*component_distances, axis=1))
            
            neighbor_reach = neighbor_distances + neighbor_radii
            
            invalid_neighbors = valid_close_index[neighbor_reach <= hole_radius]
            
            valid_idx[invalid_neighbors] = 0
            
        print("Holes filtered: ", np.count_nonzero(valid_idx==0))
        
        self.holes_xyz = self.holes_xyz[valid_idx]
        
        self.holes_radii = self.holes_radii[valid_idx]
            
            
    
    
    
    def remove_hole_intersect_data(self):
        """
        Given the array of triangle vertices in self.void_sphere_coord_data, and the hole
        parameters given by self.holes_xyz and self.hole_radii, remove all the
        triangles from self.void_sphere_coord_data who belong to one hole but live within
        a different hole
        
           ___hole1_____  ___hole2____           ___hole1_____  ___hole2____
          /             \/            \         /             \/            \
         /           -->/\ <--         \       /                             \
        |    remove-->/   \<--remove    \     |                               \
        \          -->\   /<--          /     \                               /
         \          -->\/ <--          /       \                             /
          \            /\             /         \            /\             /
           \__________/  \___________/           \__________/  \___________/
        
        Iterate through each hole in hole_xyz, find all holes who have an intersect
        with the current hole (via kdtree radius query).
        
        Since holes are spherical, any triangle whose all 3 verticies are less than
        hole_radius away from our current target means that triangle lives within
        our current hole but is not part of our current hole and can be chopped.
        """
        
        valid_vertex_idx = np.ones(self.void_sphere_coord_data.shape[0], dtype=np.uint8)
        
        ######################################################################
        # correct way to get neighbor index - 2*hole_radii.max()
        # added a distance check in union_vertex_selection to discard holes
        # who are more than hole_radius+hole_radius_neighbor apart and
        # not check those vertices
        #
        # Maybe can rework this into a cythonized method that checks against
        # hole_radius+hole_radius_neighbor up front and then we can discard
        # the 2ndary check in union_vertex_selection, not sure what's better
        #
        ######################################################################
        
        
        
        neighbor_index = self.hole_kdtree.query_radius(self.holes_xyz, 2.0*self.holes_radii.max())
        
        ######################################################################
        # Create an index of which vertices to keep via the above algorithm
        ######################################################################
        
        smooth_seams = False
        
        union_vertex_selection(neighbor_index,
                               valid_vertex_idx,
                               self.holes_xyz.astype(np.float32),
                               self.holes_radii.astype(np.float32),
                               self.void_sphere_coord_data,
                               self.vert_per_sphere,
                               smooth_seams
                               )
        
        #needed to use numpy.where maybe due to uint8 type on valid_vertex_idx?
        
        keep_idx = np.where(valid_vertex_idx)[0]
        
        self.void_sphere_coord_data = self.void_sphere_coord_data[keep_idx]
    
        self.void_sphere_normals_data = self.void_sphere_normals_data[keep_idx]
        
        self.void_sphere_coord_map = self.void_sphere_coord_map[keep_idx]
        
        
        
        

    def press_spacebar(self):
        """
        Keeping this function because it shows how to convert from the view matrix
        position to real coordinate space position
        """
        
        """
        view_camera_location = self.view[3,0:3].reshape(1,3)
        
        curr_rot_matrix = self.view[0:3,0:3]
        
        curr_camera_location = np.matmul(curr_rot_matrix, view_camera_location.T).T
        
        close_idx = self.kdtree.query_radius(-curr_camera_location, 50.0)
        
        close_idx = close_idx[0]
        
        self.vertex_data['a_bg_color'][close_idx,0] = 0.0 #R
        self.vertex_data['a_bg_color'][close_idx,1] = 1.0 #G
        self.vertex_data['a_bg_color'][close_idx,2] = 0.0 #B
        
        self.vertex_buffer.set_data(self.vertex_data)
        
        self.update()
        """
        pass
        
    def update_highlight_sphere_location(self):
        """
        If this setting is enabled, on translation updates check the new position
        for proximity to the nearest void hole, if inside that hole update the
        green (or user-colored) sphere to the size and location of the current
        void hole so the user knows they have entered a void
        
        Don't need to call self.update() cause it will be called by
        self.translate_camera() after this call
        """
        
        view_camera_location = self.view[3,0:3].reshape(1,3)
        
        curr_rot_matrix = self.view[0:3,0:3]
        
        curr_camera_location = -1.0*np.matmul(curr_rot_matrix, view_camera_location.T).T
        
        ind = self.hole_kdtree.query(curr_camera_location, k=1, return_distance=False)
        
        hole_idx = ind[0][0]
        
        hole_radius = self.holes_radii[hole_idx]
        
        hole_xyz = self.holes_xyz[hole_idx]
        
        component_dists = hole_xyz - curr_camera_location
        
        currently_inside_hole = np.sum(component_dists*component_dists) < hole_radius*hole_radius
        
        if self.highlight_state != hole_idx and currently_inside_hole:
            
            self.highlight_sphere_vertex_data["position"][:,0:3] = 0.99*hole_radius*self.unit_sphere + hole_xyz
            
            self.highlight_sphere_vertex_data["color"][:,3] = self.void_highlight_alpha
            
            self.highlight_sphere_VB.set_data(self.highlight_sphere_vertex_data)
            
            self.highlight_state = hole_idx
            
        elif self.highlight_state == hole_idx and not currently_inside_hole:
            
            self.highlight_sphere_vertex_data["color"][:,3] = 0.0
            
            self.highlight_state = -1
            
        
    def read_front_buffer(self):
        
        
        
        type_ = gloo.gl.GL_UNSIGNED_BYTE
        
        viewport = gloo.gl.glGetParameter(gloo.gl.GL_VIEWPORT)
        
        x, y, w, h = viewport
        
        #x = 0
        #y = 0
        #w = 1800
        #h = 1000
        
        gloo.gl.glPixelStorei(gloo.gl.GL_PACK_ALIGNMENT, 1)  # PACK, not UNPACK
        '''
        if mode == 'depth':
            fmt = gloo.gl.GL_DEPTH_COMPONENT
            shape = (h, w, 1)
        elif mode == 'stencil':
            fmt = gloo.gl.GL_STENCIL_INDEX8
            shape = (h, w, 1)
        elif alpha:
        
            fmt = gloo.gl.GL_RGBA
            shape = (h, w, 4)
        
        else:
            fmt = gloo.gl.GL_RGB
            shape = (h, w, 3)
        '''
        
        fmt = gloo.gl.GL_RGBA
        shape = (h, w, 4)
        
        im = gloo.gl.glReadPixels(x, y, w, h, fmt, type_)
        
        gloo.gl.glPixelStorei(gloo.gl.GL_PACK_ALIGNMENT, 4)
        
        # reshape, flip, and return
        if not isinstance(im, np.ndarray):
            
            np_dtype = np.uint8 if type_ == gloo.gl.GL_UNSIGNED_BYTE else np.float32
            
            im = np.frombuffer(im, np_dtype)
    
        im.shape = shape
        im = im[::-1, ...]  # flip the image
        
        return im.copy()
            
    def translate_camera(self, idx, plus_minus):
        
        self.view[3,idx] += plus_minus*self.translation_sensitivity
        
        self.galaxy_point_program['u_view'] = self.view
        
        self.void_sphere_program["u_view"] = self.view
        '''
        if self.enable_void_interior_highlight:
            
            self.highlight_program["u_view"] = self.view
        
            self.update_highlight_sphere_location()
        '''
        
        '''
        self.update()
        
        if self.recording:
            
            #img = self.render().copy()
            
            img = self.read_front_buffer()
        
            img[:,:,3] = 255
            
            self.recording_frames.append(img)
        '''
        #app.Canvas.update(self)
        
        
        
    def rotate_camera(self, idx, plus_minus):
        '''
        Helper function to control pitch, yaw, and roll
        of the camera.
        '''
        
        axis_vector = [0.0, 0.0, 0.0]
        
        axis_vector[idx] = plus_minus
        
        curr_rotation = rotate(self.rotation_sensitivity, axis_vector)
        
        self.view = np.matmul(self.view, curr_rotation)
        
        self.galaxy_point_program['u_view'] = self.view
        
        self.void_sphere_program["u_view"] = self.view
        '''
        if self.enable_void_interior_highlight:
            
            self.highlight_program["u_view"] = self.view
        '''
        
        
        '''
        self.update()
        
        if self.recording:
            
            #img = self.render().copy()
            
            img = self.read_front_buffer()
        
            img[:,:,3] = 255
            
            self.recording_frames.append(img)
        '''
        #app.Canvas.update(self)
        
        
        
    def press_w(self):
        
        self.translate_camera(2, 1.0)
        
    def press_s(self):
        
        self.translate_camera(2, -1.0)
        
    def press_a(self):
        
        self.translate_camera(0, 1.0)
        
    def press_d(self):
        
        self.translate_camera(0, -1.0) 
        
    def press_r(self):
        
        self.translate_camera(1, -1.0)
        
    def press_f(self):
        
        self.translate_camera(1, 1.0) 
        
    def press_z(self):
        
        self.translation_sensitivity *= 1.1
        
    def press_x(self):
        
        self.translation_sensitivity /= 1.1
        
    def press_c(self):
        
        self.rotation_sensitivity *= 1.1
        
    def press_v(self):
        
        self.rotation_sensitivity /= 1.1
        
    def press_i(self):
        
        self.rotate_camera(0, -1.0)
        
    def press_k(self):
        
        self.rotate_camera(0, 1.0)
        
    def press_j(self):
        
        self.rotate_camera(1, -1.0)
        
    def press_l(self):
        
        self.rotate_camera(1, 1.0)
        
    def press_q(self):
        
        self.rotate_camera(2, -1.0)
        
    def press_e(self):
                
        self.rotate_camera(2, 1.0)
        
    def press_p(self):
        """
        Helper function to print current camera location
        """
        
        curr_camera_location = self.view[3,0:3].reshape(1,3)
        
        curr_rot_matrix = self.view[0:3,0:3]
        
        data_coord_camera_location = -1.0*np.matmul(curr_rot_matrix, curr_camera_location.T).T
        
        norm = np.sqrt(np.sum(curr_camera_location*curr_camera_location))
        
        #Print the current view matrix, actual camera location, and distance to origin
        print("View info: ")
        print(self.view)
        print(data_coord_camera_location)
        print(norm)
        
    def press_m(self):
        
        print("Pressed m!")
        
        if self.recording:
            
            print("Writing frames...")
            
            #Write to png and save video
            
            random_ID = self.random_string()
            
            ffmpeg_list_filename = random_ID+".txt"
            
            ffmpeg_list_file = open(ffmpeg_list_filename, 'wb')
            
            ffmpeg_file_list = []
            
            for idx, frame in enumerate(self.recording_frames):
                
                outname = random_ID + "_frame_" + str(idx) + ".png"
                
                ffmpeg_file_list.append(outname)
                
                outpath = os.path.join(os.getcwd(), outname)
                
                ffmpeg_list_file.write("file ".encode("utf-8")+outpath.encode('utf-8')+"\n".encode('utf-8'))
        
                io.write_png(outname, frame)
                
            ############################################################
            # Use FFMPEG to convert from frames to an mp4
            ############################################################
            
            print("FFMPEG compiling video...")
            
            #command = "ffmpeg -r 10 -f concat -safe 0 -i "+ffmpeg_list_filename+' -c:v libx264 -vsync vfr -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p '+random_ID+".mp4"
            
            
            #print(command)
            
            command = ["ffmpeg", 
                       "-r", "25",
                       "-f", "concat",
                       "-safe", "0",
                       "-i", ffmpeg_list_filename,
                       "-c:v", "libx264",
                       "-vsync", "vfr",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       "-pix_fmt", "yuv420p",
                       random_ID+".mp4"]
            
            #proc = subprocess.Popen(["ffmpeg", "-i", ffmpeg_list_filename, "-c:v libx264", random_ID+".mp4"])
            #proc = subprocess.Popen(command, shell=True)
            proc = subprocess.Popen(command)
            
            exit_code = proc.wait()
            
            ############################################################
            # Clean up the memory and disk resources
            ############################################################
            
            print("Cleaning disk...")
            
            self.recording_frames = []
            
            for curr_name in ffmpeg_file_list:
                
                os.remove(curr_name)
                
            os.remove(ffmpeg_list_filename)
            
            ############################################################
            # All done
            ############################################################
            print("Finished!")
            
            self.recording = False
        
        else:
            
            #Start recording
            
            self.recording = True
            
            pass
        
        
        
    def press_0(self):
        
        img = self.render().copy()
        
        print(img.shape)
        
        print(img[500,900])
        
        img[:,:,3] = 255
        
        random_sequence = self.random_string()
        
        outname = "voidfinder_" + random_sequence + ".png"
        
        io.write_png(outname, img)
        
    def random_string(self, num=6):
        
        alpha_num = "abcdefghijklmnopqrstuvwxyz1234567890"
        
        random_sequence = ""
        
        for _ in range(num):
            
            curr_char = alpha_num[np.random.randint(0,36)]
            
            random_sequence += curr_char
            
        return random_sequence
        

    def on_key_press(self, event):
        """
        Activated when user pressed a key on the keyboard
        """
        
        #print(vars(event))
        
        if event.text in self.keyboard_commands:
            
            self.keyboard_active[event.text] = 1
            
        elif event.text in self.press_once_commands:
            
            self.press_once_commands[event.text]()
            
        for curr_key in self.keyboard_active:
            
            if self.keyboard_active[curr_key]:
    
                self.keyboard_commands[curr_key]()
            
            
    def on_key_release(self, event):
        """
        Activates when user releases a key
        """
        
        if event.text in self.keyboard_commands:
            
            self.keyboard_active[event.text] = 0
    

    def on_timer(self, event):
        """
        Callback every .01 seconds or so, mostly to process keyboard 
        commands for now
        """
        if time.time() - self.last_keypress_time > 0.02:
            
            for curr_key in self.keyboard_active:
            
                if self.keyboard_active[curr_key]:
                    
                    self.keyboard_commands[curr_key]()
                    
        self.update()
        
        if self.recording:
            
            #img = self.render().copy()
            
            img = self.read_front_buffer()
        
            img[:,:,3] = 255
            
            self.recording_frames.append(img)
        
        
        
    def on_resize(self, event):
        
        self.apply_zoom()
        

    def on_mouse_wheel(self, event):
        """
        Make the galaxies (displayed as gl_PointCoords discs) display radius
        larger or smaller up to max size
        """
    
        self.point_size_denominator -= event.delta[1]
        
        self.point_size_denominator = max(1.0, self.point_size_denominator)
        
        self.point_size_denominator = min(10.0, self.point_size_denominator)
    
        self.galaxy_point_program['u_size'] = 1.0 / self.point_size_denominator
        
        self.update()
        
        
    def on_mouse_press(self, event):
        
        if self.mouse_state == 0:
            
            self.mouse_state = 1
            
            self.last_mouse_pos = event.pos


    def on_mouse_release(self, event):
        
        if self.mouse_state == 1:
            
            self.mouse_state = 0


    def on_mouse_move(self, event):
        
        if self.mouse_state == 1 and event.button == 1:
            
            #print(vars(event))
            
            curr_x = event.pos[0]
            curr_y = event.pos[1]
            
            
            curr_x_delta = self.last_mouse_pos[0] - curr_x
            curr_y_delta = self.last_mouse_pos[1] - curr_y
            '''
            print(self.last_mouse_pos, event.pos)
            
            print("Last y - curr y: ", self.last_mouse_pos[1], curr_y)
            print("Last x - curr x: ", self.last_mouse_pos[0], curr_x)
            
            print("Curr x delta: ", curr_x_delta)
            print("Curr y delta: ", curr_y_delta)
            '''
            self.last_mouse_pos = event.pos
            
            if abs(curr_x_delta) > abs(curr_y_delta):
                
                curr_y_delta = 0
                
            else:
                
                curr_x_delta = 0
            
            if curr_y_delta < 0:
                
                self.rotate_camera(0, -1.0)
                
            elif curr_y_delta > 0:
                
                self.rotate_camera(0, 1.0)
                
            if curr_x_delta < 0:
                
                self.rotate_camera(1, -1.0)
                
            elif curr_x_delta > 0:
                
                self.rotate_camera(1, 1.0)
            
            
            pass
            
        elif self.mouse_state == 1 and event.button == 2:
            
            #curr_x = event.pos[0]
            curr_y = event.pos[1]
            
            
            #curr_x_delta = self.last_mouse_pos[0] - curr_x
            curr_y_delta = self.last_mouse_pos[1] - curr_y
            
            self.last_mouse_pos = event.pos
            
            if curr_y_delta < 0:
                
                self.translate_camera(2, 1.0)
                
            elif curr_y_delta > 0:
                
                self.translate_camera(2, -1.0)
            
            
    def on_draw(self, event):
        
        gloo.clear((1, 1, 1, 1))
        
        self.galaxy_point_program.draw('points')
        
        self.void_sphere_program.draw('triangles')
        '''
        if self.enable_void_interior_highlight:
        
            self.highlight_program.draw('triangles')
        '''

    def apply_zoom(self):
        """
        For a fov angle of 60 degrees and a near plane of 0.01, 
        the size of the near viewing plane is .0115 in height and .0153 in width (on 800x600 screen)
        """
        
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        
        self.projection = perspective(60.0, self.size[0]/float(self.size[1]), 0.01, 10000.0)
        
        self.galaxy_point_program['u_projection'] = self.projection
        
        self.void_sphere_program['u_projection'] = self.projection
        '''
        if self.enable_void_interior_highlight:
            
            self.highlight_program['u_projection'] = self.projection
        '''
    def run(self):
        app.run()


if __name__ == "__main__":
    
    holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")
    
    galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
    #galaxy_data = load_galaxy_data('kias1033_5.dat')
    #galaxy_data = load_galaxy_data("dr12n.dat")
    
    print("Holes: ", holes_xyz.shape, holes_radii.shape, holes_flags.shape)
    
    hole_IDs = np.unique(holes_flags)
    
    num_hole_groups = len(hole_IDs)
    
    from vispy.color import Colormap
    
    cm = Colormap(['#880000',
                   '#EEEE00',
                   "#008800",
                   '#EE00EE',
                   '#000088',
                   '#EE00EE'])
    
    hole_color_vals = cm.map(np.linspace(0, 1.0, num_hole_groups))
    
    print(hole_color_vals.shape)
    
    void_hole_colors = np.empty((holes_xyz.shape[0],4), dtype=np.float32)
    
    for idx in range(void_hole_colors.shape[0]):
        
        hole_group = holes_flags[idx] 
        
        #print(hole_group)
        
        void_hole_colors[idx,:] = hole_color_vals[hole_group-1] #uhg you used 1-based indexing WHY? :D
            
    
    
    
    print("Galaxies: ", galaxy_data.shape)
    
    viz = VoidRender(holes_xyz, 
                          holes_radii, 
                          galaxy_data,
                          galaxy_display_radius=10,
                          #void_hole_color=np.array([0.0, 0.0, 1.0, 0.95], dtype=np.float32),
                          void_hole_color=void_hole_colors,
                          SPHERE_TRIANGULARIZATION_DEPTH=4,
                          canvas_size=(1600,1200))
    
    viz.run()
    #app.run()
    
    
    