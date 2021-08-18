"""Void rendering program based on OpenGL and vispy.
"""

import os
import subprocess

import numpy as np

from vispy import app, gloo
import vispy.io as io
from vispy.util.transforms import perspective, translate, rotate

#from vispy.color import Color

from scipy.spatial import cKDTree

from sklearn import neighbors

import time

#import gc

#Vertex shader for the Galaxy Vispy Program
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
#Fragment shader for the Galaxy Vispy Program
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


#Vertex shader for the Void Sphere Vispy Program
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

#Fragment shader for the Void Sphere Vispy Program
frag_sphere = """

#version 120

varying vec4 v_color;

void main()
{

    gl_FragColor = v_color;
    //gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

"""



vert_meridians = """

#version 120

uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec4 a_position;

void main()
{
    gl_Position = u_projection * u_view * a_position;
}


"""

frag_meridians = """

#version 120


void main()
{

    //gl_FragColor = v_color;
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}

"""


class Triangle(object):
    """
    Simple helper class for icosahedron sphere triangularization.
    """
    def __init__(self, pt1, pt2, pt3):
        
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3





class VoidRender(app.Canvas):

    def __init__(self,
                 voids_tri_x=None,
                 voids_tri_y=None,
                 voids_tri_z=None,
                 voids_norm=None,
                 voids_id=None,
                 galaxy_xyz=None,
                 galaxy_display_radius=2.0,
                 gal_viz=None,
                 gal_opp=None,
                 canvas_size=(800,600),
                 title="VoidFinder Results",
                 camera_start_location=None,
                 camera_start_orientation=None,
                 start_translation_sensitivity=1.0,
                 start_rotation_sensitivity=1.0,
                 galaxy_color=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                 void_color=np.array([0.0, 0.0, 1.0, 0.95], dtype=np.float32)
                 ):
        '''
        Main class for initializing the visualization.
        
        Examples
        ========
        
        from vast.vsquared.viz import VoidRender, load_void_data, load_galaxy_data
        
        voids_tri_x, voids_tri_y, voids_tri_z, voids_norm, voids_id, gal_viz, gal_opp = load_void_data("DR7_triangles.dat", "DR7_galviz.dat")
    
        galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.fits")
    
        viz = VoidFinderCanvas(voids_tri_x, voids_tri_y, voids_tri_z, 
                               voids_norm, 
                               voids_id, 
                               galaxy_data,
                               gal_viz, 
                               gal_opp, 
                               canvas_size=(1600,1200))
    
        viz.run()
        

        Notes
        =====
        
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
        ==========
        
        holes_xyz : (N,3) numpy.ndarray
            xyz coordinates of the hole centers
            
        holes_radii : (N,) numpy.ndarray
            length of the hole radii in xyz coordinates
            
        holes_group_IDs : (N,) numpy.ndarray of integers
            Void group to which a given hole belongs according to VoidFinder
            
        galaxy_xyz : (N,3) numpy.ndarray
            xyz coordinates of the galaxy locations
            
        galaxy_display_radius : float
            using a constant radius to display galaxy points since they should
            all be small compared to the void holes, and they don't have
            corresponding radii
            
        remove_void_intersects : int, default 1
            0 - turn off
            1 - remove all intersections
            2 - remove intersections only within predefined Void Groups based on hole_group_IDs
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
        
        '''
        if holes_xyz is None and galaxy_xyz is None:
            raise ValueError("holex_xyz and galaxy_xyz cannot both be None, or there's nothing to display!")
        '''
        
        
        app.Canvas.__init__(self, 
                            keys='interactive', 
                            size=canvas_size)
        
        #self.measure_fps()
        
        self.title = title
        
        self.translation_sensitivity = start_translation_sensitivity
        
        self.rotation_sensitivity = start_rotation_sensitivity
        
        self.max_galaxy_display_radius = galaxy_display_radius

        self.unit_sphere, self.unit_sphere_normals = self.create_sphere(1.0, 2)
        
        ######################################################################
        # Allow user to show just voids of just galaxies
        ######################################################################
        
        self.voids_enabled = voids_tri_x is not None
        
        self.galaxies_enabled = galaxy_xyz is not None

        self.enable_void_highlight = False
        
        self.enabled_programs = []
        
        ######################################################################
        #
        ######################################################################
        if self.voids_enabled:
            
            self.voids_tri_x = voids_tri_x
            
            self.voids_tri_y = voids_tri_y
            
            self.voids_tri_z = voids_tri_z
        
            self.voids_norm = voids_norm
            
            self.voids_id = voids_id
            
            self.void_color = void_color
            
            self.num_void = len(np.unique(voids_id))

            if gal_viz is not None:
                self.gal_kdtree = cKDTree(galaxy_xyz[gal_viz != -1])
                self.gal2void = gal_viz[gal_viz != -1]
                self.gal2opp = gal_opp[gal_viz != -1]
                self.enable_void_highlight = True
            
            self.setup_void_program()
        
        ######################################################################
        #
        ######################################################################
        if self.galaxies_enabled:
            
            self.galaxy_xyz = galaxy_xyz
        
            self.num_gal = galaxy_xyz.shape[0]
        
            self.galaxy_color = galaxy_color
            
            self.setup_galaxy_program()
        
        
        ######################################################################
        #
        ######################################################################

        #self.setup_orientation_sphere_program()
        
        
        ######################################################################
        #
        ######################################################################
        
        
        self.setup_camera_view(camera_start_location, camera_start_orientation)
        
        if self.enable_void_highlight:
            self.setup_highlight_program()
        
        self.setup_mouse()
        
        self.setup_keyboard()
        
        self.script_running = False
        
        self.script = []
        
        self.script_idx = 0
        
        self.recording = False
        
        self.recording_frames = []
        
        ######################################################################
        #
        ######################################################################
        
        
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
                                  'e' : self.press_e,
                                  '-' : self.mouse_wheel_down,
                                  '+' : self.mouse_wheel_up}
        
        self.press_once_commands = {" " : self.press_spacebar,
                                    "p" : self.press_p,
                                    "m" : self.press_m,
                                    "0" : self.press_0,
                                    "n" : self.press_n}
        
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
            '''        
            if self.holes_enabled:
                start_location = np.mean(self.holes_xyz, axis=0)
            else:
                start_location = np.mean(self.galaxy_xyz, axis=0)
            '''
            start_location = np.mean(self.galaxy_xyz, axis=0)

            start_location[2] += 300.0
            
            start_location *= -1.0
        
        self.view = np.eye(4, dtype=np.float32)
        
        self.view[3,0:3] = start_location
        
        if start_orientation is not None:
            
            self.view[0:3,0:3] = start_orientation
            
        self.projection = np.eye(4, dtype=np.float32) #start with orthographic, will modify this
        
        
        
        

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
        
        #self.galaxy_point_program['u_view'] = self.view
        
        self.galaxy_point_program['u_size'] = 1.0
        
        self.enabled_programs.append((self.galaxy_point_program, "points"))

        
        
    def setup_void_program(self):
        """
        This vispy program draws all the triangles which compose the spheres
        representing the void holes
        """
        
        
        ######################################################################
        # Initialize some space to hold all the vertices (and w coordinate)
        # for all the vertices of all the spheres for all the void holes
        ######################################################################
        
        #print("Creating sphere memory")
        
        #num_sphere_verts = self.vert_per_sphere*self.holes_xyz.shape[0]
        
        #num_sphere_triangles = num_sphere_verts//3
        
        #self.void_coord_data = np.ones((num_sphere_verts,4), dtype=np.float32)

        self.void_coord_data = np.array([self.voids_tri_x.reshape(3*len(self.voids_tri_x)),self.voids_tri_y.reshape(3*len(self.voids_tri_y)),self.voids_tri_z.reshape(3*len(self.voids_tri_z))]).tolist()
        self.void_coord_data.append(np.ones(len(self.void_coord_data[0]), dtype=np.float32).tolist())
        self.void_coord_data = np.array(self.void_coord_data, dtype=np.float32).T
        
        self.void_coord_map = np.repeat(self.voids_id, 3)
        
        #self.void_normals_data = np.zeros((num_sphere_verts,4), dtype=np.float32)

        self.void_normals_data = np.repeat(self.voids_norm,3,axis=0).T.tolist()
        self.void_normals_data.append(np.ones(len(self.void_normals_data[0]), dtype=np.float32).tolist())
        self.void_normals_data = np.array(self.void_normals_data, dtype=np.float32).T        
        
        ######################################################################
        # Calculate all the sphere vertex positions and add them to the
        # vertex array, and the centroid of each triangle
        # ERRRT - don't need centroids, they're close enough to the vertices
        # themselves, just use the vertex positions, and a 
        # a copy of the normals!
        ######################################################################
        
        #print("Calculating sphere positions")
        '''
        for idx, (hole_xyz, hole_radius) in enumerate(zip(self.holes_xyz, self.holes_radii)):
            
            #curr_sphere = (self.unit_sphere * hole_radius) + hole_xyz
            
            start_idx = idx*self.vert_per_sphere
            
            end_idx = (idx+1)*self.vert_per_sphere
            
            #self.void_sphere_coord_data[start_idx:end_idx, 0:3] = curr_sphere
            
            self.void_sphere_coord_data[start_idx:end_idx, 0:3] = (self.unit_sphere * hole_radius) + hole_xyz
            
            self.void_sphere_coord_map[start_idx:end_idx] = idx
            
            self.void_sphere_normals_data[start_idx:end_idx, 0:3] = self.unit_sphere_normals
            
            #for jdx in range(num_sphere_triangles):
                
                #self.void_sphere_centroid_data[kdx] = np.mean(curr_sphere[jdx:(jdx+3)], axis=0)
                
            #gc.collect()
                
        '''   
        ######################################################################
        # Given there will be a lot of overlap internal to the spheres, 
        # remove the overlap for better viewing quality
        ######################################################################
        '''
        if self.remove_void_intersects > 0:
            
            print("Pre intersect-remove vertices: ", self.void_sphere_coord_data.shape[0])
            
            start_time = time.time()
            
            self.remove_hole_intersect_data()
            
            remove_time = time.time() - start_time
            
            num_sphere_verts = self.void_sphere_coord_data.shape[0]
            
            print("Post intersect-remove vertices: ", num_sphere_verts, "time: ", remove_time)
            
        '''
        ######################################################################
        #
        ######################################################################
        
        self.void_vertex_data = np.zeros(self.void_coord_data.shape[0], [('position', np.float32, 4),
                                                                   ('normal', np.float32, 4),
                                                                   ('color', np.float32, 4)])
        
        self.void_vertex_data["position"] = self.void_coord_data
        
        self.void_vertex_data["normal"] = self.void_normals_data
        
        
        ######################################################################
        # Set up the colors for the holes
        ######################################################################
        
        if self.void_color.shape[0] == self.num_void:
            
            print("Coloring based on self.void_hole_color of shape: ", self.void_color.shape)
            
            void_colors = np.empty((self.void_coord_data.shape[0], 4), dtype=np.float32)
            
            for idx, void_idx in enumerate(self.void_coord_map):
                
                #vindx = np.unique(self.voids_id,return_index=True)[1]
                #void_idy = np.where([self.voids_id[vind] for vind in sorted(vindx)]==void_idx)[0][0]
                void_colors[idx,:] = self.void_color[void_idx,:]
                
            print(void_colors.min(), void_colors.max())
            
        else:
            
            print(self.void_color.shape, self.num_void)
            print("Coloring all voids same color")
        
            void_colors = np.tile(self.void_color, (self.void_color.shape[0], 1))
            
            print(void_colors.min(), void_colors.max())
        
        self.void_vertex_data["color"] = void_colors
        
        
        
        ######################################################################
        # Set up the sphere-drawing program
        ######################################################################
        
        print("Void Sphere program vertices: ", self.void_vertex_data["position"].shape)
        
        self.void_VB = gloo.VertexBuffer(self.void_vertex_data)
        
        self.void_sphere_program = gloo.Program(vert_sphere, frag_sphere)
        
        self.void_sphere_program.bind(self.void_VB)
        
        #self.void_sphere_program['u_view'] = self.view
        
        self.enabled_programs.append((self.void_sphere_program, "triangles"))
        
        
        
        
    def setup_orientation_sphere_program(self):
        """
        One program for the meridians since we can be tricky and plot them in
        such a way as to use the openGL "line_strip" draw type.
        
        One program for the parallels since they have to be kinda standalone
        but I don't want to write 1 "line_loop" program per parallel, so
        I'm going to write 1 "lines" draw type program for all the parallels
        
        """
        ######################################################################
        # Calculate the radius to make the sphere
        ######################################################################
        
        orientation_sphere_radius = 500.0
        
        
        
        ######################################################################
        # We can draw the meridians (lines of constant longitude) as an
        # OpenGL "line_loop" by using the fact that all meridians intersect
        # at the north pole and south pole.
        #
        # From the starting longitude, draw from north pole to south pole, 
        # turn by the longitude increment, then draw the next line from
        # south pole to north pole, turn again, repeat north to south, etc.
        ######################################################################
        
        num_meridians = 20
        
        meridian_depth = 200
        
        meridian_latitudes = np.linspace(np.pi/2.0, -np.pi/2.0, meridian_depth)
        
        meridian_longitudes = np.linspace(0.0, 2.0*np.pi, num_meridians+1) #linspace includes the endpoint so +1 num_meridians
        
        lat_sines = np.sin(meridian_latitudes)
        lat_coses = np.cos(meridian_latitudes)
        long_sines = np.sin(meridian_longitudes)
        long_coses = np.cos(meridian_longitudes)
        
        meridian_coords = np.empty((num_meridians*meridian_depth, 3), dtype=np.float32)
        
        flipper = -1
        
        out_idx = 0
        
        for idx in range(num_meridians): 
            
            if flipper == -1:
                lat_iter = range(meridian_depth)
                flipper = 1
            elif flipper == 1:
                lat_iter = reversed(range(meridian_depth))
                flipper = -1
                
            for jdx in lat_iter:
                
                y = lat_sines[jdx]
                z = lat_coses[jdx]*long_sines[idx]
                x = lat_coses[jdx]*long_coses[idx]
                
                meridian_coords[out_idx,0] = x
                meridian_coords[out_idx,1] = y
                meridian_coords[out_idx,2] = z
                
                out_idx += 1
                
        
        meridian_coords *= orientation_sphere_radius
        
        self.orientation_sphere_meridians = np.ones(meridian_coords.shape[0], 
                                                     [
                                                      ('a_position', np.float32, 4),
                                                      #('a_bg_color', np.float32, 4),
                                                      #('a_fg_color', np.float32, 4),
                                                      #('a_size', np.float32)
                                                     ]
                                                     )
        
        #w_col = np.ones((self.num_gal, 1), dtype=np.float32)
        
        self.orientation_sphere_meridians['a_position'][:,0:3] = meridian_coords
        
        #self.galaxy_vertex_data['a_bg_color'] = np.tile(self.galaxy_color, (self.num_gal,1))
        
        #black = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) #black color
        
        #self.galaxy_vertex_data['a_fg_color'] = np.tile(black, (self.num_gal,1)) 
        
        #self.galaxy_vertex_data['a_size'] = self.max_galaxy_display_radius #broadcast to whole array?
        
        self.orientation_sphere_meridians_VB = gloo.VertexBuffer(self.orientation_sphere_meridians)
        
        self.orientation_sphere_meridian_program = gloo.Program(vert_meridians, frag_meridians)
        
        self.orientation_sphere_meridian_program.bind(self.orientation_sphere_meridians_VB)
        
        self.enabled_programs.append((self.orientation_sphere_meridian_program, "line_strip"))
        
        
        ######################################################################
        # Next do lines of constant latitude. 
        ######################################################################
        
        num_parallels = 20
        
        parallel_depth = 200
        
        parallel_latitudes = np.linspace(np.pi/2.0, -np.pi/2.0, num_parallels+2)[1:-1] #exclude north and south poles
        
        parallel_longitudes = np.linspace(0.0, 2.0*np.pi, parallel_depth+1) #linspace includes the endpoint so +1 num_meridians
        
        lat_sines = np.sin(parallel_latitudes)
        lat_coses = np.cos(parallel_latitudes)
        long_sines = np.sin(parallel_longitudes)
        long_coses = np.cos(parallel_longitudes)
        
        parallel_coords = np.empty((2*num_parallels*parallel_depth, 3), dtype=np.float32)
        
        out_idx = 0
        
        for jdx in range(num_parallels):
            
            for idx in range(parallel_depth):
                
                y = lat_sines[jdx]
                z = lat_coses[jdx]*long_sines[idx]
                x = lat_coses[jdx]*long_coses[idx]
                
                y2 = lat_sines[jdx]
                z2 = lat_coses[jdx]*long_sines[idx+1]
                x2 = lat_coses[jdx]*long_coses[idx+1] 
                
                parallel_coords[out_idx,0] = x
                parallel_coords[out_idx,1] = y
                parallel_coords[out_idx,2] = z
                
                out_idx += 1
                
                parallel_coords[out_idx,0] = x2
                parallel_coords[out_idx,1] = y2
                parallel_coords[out_idx,2] = z2
                
                out_idx += 1
                
                
        
        
        parallel_coords *= orientation_sphere_radius
        
        self.orientation_sphere_parallels = np.ones(parallel_coords.shape[0], 
                                                     [
                                                      ('a_position', np.float32, 4),
                                                      #('a_bg_color', np.float32, 4),
                                                      #('a_fg_color', np.float32, 4),
                                                      #('a_size', np.float32)
                                                     ]
                                                     )
        
        #w_col = np.ones((self.num_gal, 1), dtype=np.float32)
        
        self.orientation_sphere_parallels['a_position'][:,0:3] = parallel_coords
        
        #self.galaxy_vertex_data['a_bg_color'] = np.tile(self.galaxy_color, (self.num_gal,1))
        
        #black = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) #black color
        
        #self.galaxy_vertex_data['a_fg_color'] = np.tile(black, (self.num_gal,1)) 
        
        #self.galaxy_vertex_data['a_size'] = self.max_galaxy_display_radius #broadcast to whole array?
        
        self.orientation_sphere_parallels_VB = gloo.VertexBuffer(self.orientation_sphere_parallels)
        
        self.orientation_sphere_parallels_program = gloo.Program(vert_meridians, frag_meridians)
        
        self.orientation_sphere_parallels_program.bind(self.orientation_sphere_parallels_VB)
        
        self.enabled_programs.append((self.orientation_sphere_parallels_program, "lines"))
        
                
        
        

        
    def setup_highlight_program(self):
        """
        Setup the sphere for when a user enters a void so it colors it
        green (or a different user-specified color) so you know you're 
        inside a void
        """
        
        self.current_state = 0

        self.current_void_coord_data = self.void_coord_data[self.void_coord_map==self.current_state]
        
        #self.current_void_coord_data = np.ones(((self.void_coord_data[self.void_coord_map==self.current_state]).shape[0],4), dtype=np.float32)
        
        #self.current_void_coord_data[:,0:3] = 10.0*self.unit_sphere
        
        ######################################################################
        #
        ######################################################################
        self.current_void_vertex_data = np.zeros((self.void_coord_data[self.void_coord_map==self.current_state]).shape[0], [('position', np.float32, 4),
                                                                                 ('color', np.float32, 4)])
        
        self.current_void_vertex_data["position"] = self.current_void_coord_data

        self.current_void_vertex_data["color"] = self.void_vertex_data["color"][self.void_coord_map==self.current_state]
        
        #self.current_void_vertex_data["color"] = np.tile(self.void_color[:self.unit_sphere.shape[0]], (self.current_void_coord_data.shape[0], 1))
        
        self.current_void_vertex_data["color"][:,3] = 0.0
        
        self.current_void_VB = gloo.VertexBuffer(self.current_void_vertex_data)
        
        ######################################################################
        #
        ######################################################################
        self.highlight_program = gloo.Program(vert_sphere, frag_sphere)

        self.highlight_program.bind(self.current_void_VB)
        
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
        
    def update_current_void_location(self):
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
        
        ind = self.gal_kdtree.query(curr_camera_location, k=1)
        
        gal_idx = ind[1][0]

        void_idx = self.gal2void[gal_idx]

        self.current_void_vertex_data = np.zeros((self.void_coord_data[self.void_coord_map==void_idx]).shape[0], [('position', np.float32, 4),
                                                                                 ('color', np.float32, 4)])

        if self.current_state != void_idx:

            self.current_void_vertex_data["position"] = self.void_vertex_data["position"][self.void_coord_map==void_idx]

            self.current_void_vertex_data["color"] = self.void_vertex_data["color"][self.void_coord_map==void_idx]

            self.current_void_VB.set_data(self.current_void_vertex_data)

            self.current_state = void_idx

        '''        
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
        '''
            
        
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
        
        #self.galaxy_point_program['u_view'] = self.view
        
        #self.void_sphere_program["u_view"] = self.view

        if self.enable_void_highlight:
            
            self.highlight_program["u_view"] = self.view
        
            self.update_current_void_location()
        
        
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
        
        #self.galaxy_point_program['u_view'] = self.view
        
        #self.void_sphere_program["u_view"] = self.view
        
        
        if self.enable_void_highlight:
            
            self.highlight_program["u_view"] = self.view
        
        
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
        
        
    def press_n(self):
        """
        Run a pre-defined script that takes you on a tour through the universe
        """
        
        
        if self.script_running:
            
            pass
        
        else:
            
            start_location = np.mean(self.holes_xyz, axis=0)
        
            start_location[2] += 600.0
            
            start_location *= -1.0
            
            self.translation_sensitivity = 1.0
        
            self.rotation_sensitivity = 1.0
            
            self.point_size_denominator = 1.0
            
            
            self.setup_camera_view(start_location)
            
            set_1 = [
                    [('+')]*5,
                    [('x')]*2,
                    [('w')]*1,
                    [('s')]*1,
                    [("_")]*60,
                    [('w')]*450,
                    [('w', '-')],
                    [('w', 'x', 'v')]*3,
                    [('w', '-')],
                    [('w')]*25,
                    [('w', 'x')]*3,
                    [('w')]*25,
                    [('w', '-')],
                    [('w', 'v')]*2,
                    [('w', 'j')]*90,
                    [('w')]*10,
                    [('w', 'x')]*3,
                    [('w', 'c')]*2,
                    [('w', 'e')]*80,
                    [('w', 'x')]*5,
                    [('w', 'v', 'x')]*9,
                    [('w', 'i')]*325,
                    [('w', '-')],
                    [('w', 'j', 'k')]*50,
                    [('w', 'q')]*150,
                    [('w', 'i')]*100,
                    [('w')]*100,
                    [('w', 'q')]*150,
                    [('w', 'i')]*80,
                    [('w', 'z')]*5,
                    [('w')]*150,
                    [('w', 'q', 'k')]*150,
                    [('w', 'c')]*5,
                    [('w', 'q', 'i')]*150,
                    [('w', 'i')]*30,
                    [('w')]*100,
                    [('w', 'v')]*10,
                    [('w', 'k')]*100,
                    [('w')]*200,
                    [('w', 'c')]*10,
                    [('w', 'z')],
                    [('w', 'e', 'i')]*100,
                    [('w', 'x')]*5,
                    [('w', 'a', 'l')]*65,
                    [('w', 'z')]*15,
                    [('w')]*75,
                    [('w', 'x')]*10,
                    [('w', 'd', 'j')]*100,
                    [('d', 'j', 'v')]*10,
                    [('s', 'd', 'j')]*300,
                    [('w', 'v')]*10,
                    [('w', 'k', 'l')]*150,
                    [('w', 'z')]*5,
                    [('w', 'd', 'j')]*100,
                    [('w', 'k', 'j')]*50,
                    [('w', 'j')]*25,
                    [('w', 'z')]*5,
                    [('w')]*42,
                    [('w', 'c')]*15,
                    [('w', 'j', 'i')]*90,
                    [('w', 'j', 'q')]*35,
                    [('w', 'j')]*25,
                    [('w')]*25,
                    [('w', 'c')]*5,
                    [('w', 'j', 'q', 'i')]*50,
                    [('w', 'i')]*50,
                    [('w', 'e')]*20,
                    [('w', 'e', 'i')]*50,
                    [('w', 'e', 'i', 'l')]*50,
                    [('w', 'k', 'c')]*5,
                    [('w', 'k')]*45,
                    [('w', 'k', 'c')]*5,
                    [('w', 'k')]*45,
                    [('w', 'v', 'x')]*10,
                    [('w', 'i', 'x')]*10,
                    [('w', 'i', 'v', 'x')]*10,
                    [('w', 'i')]*15,
                    [('w', 'v')]*5,
                    [('w', 'z')]*10,
                    [('w')]*35,
                    [('w', 'a', 'r', 'l')]*200,
                    [('a', 'r', 'l')]*300,
                    [('a', 'r', 'l', 'k')]*300,
                    [('a', 's', 'l', 'k')]*300,
                    [('s', 'z')]*10,
                    [('s')]*100,
                    [('s', 'z')]*10,
                    [('s', 'c')]*8,
                    [('s', 'i')]*100,
                    [('s', '+')],
                    [('s')]*25,
                    [('s', '+')],
                    [('s')]*50,
                    [('s', 'z')]*15,
                    [('s', '+')],
                    [('s')]*125,
                    [('s', '+')],
                    [('s')]*25,
                    
                    ]
            
            self.script = []
            
            for element in set_1:
            
                self.script.extend(element)
            
            print("Len script: ", len(self.script))
            
            self.script_idx = 0
            
            self.script_running = True
        
        
    def press_m(self):
        
        print("Pressed m!")
        
        if self.recording:
            
            print("Writing frames... ("+str(len(self.recording_frames))+")")
            
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
                       "-r", "60",
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
            
            
        
        
        
    def press_0(self):
        
        #img = self.render().copy()
        
        img = self.read_front_buffer()
        
        #print(img.shape)
        
        #print(img[500,900])
        
        """
        Fix the alpha compositing using:
        https://en.wikipedia.org/wiki/Alpha_compositing
        
        Color_out = color_foreground + color_background*(1- alpha_foreground)
        
        IF the raster is premultiplied alpha color.  Otherwise gotta use the more complex one.
        
        
        """
        
        #if not hasattr(self, background_color_array):
            
        #    self.background_color_array = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #    self.background_color_array.fill(255) #white
        
        
        #partial_transparency_index = img[:,:,3] < 255
        
        #print(partial_transparency_index.shape, np.sum(partial_transparency_index))
        
        #mod_percentages = 1.0 - img[partial_transparency_index,3]/255.0
        
        #mod_values = 255*mod_percentages
        
        #(255*(1 - x/255)) = 255 - x
        
        #composite_modifier_values = 255 - img[partial_transparency_index,3]
        
        #print(composite_modifier_values.shape)
        
        
        
        
        #img[:,:,3] = 255
        
        #img[partial_transparency_index,:] += composite_modifier_values
        
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
    
    
    def yield_next_script_commands(self):
        
        for element in self.script:
            
            print(element)
            
            yield element
            
        return None
        

    def on_timer(self, event):
        """
        Callback every .01 seconds or so, mostly to process keyboard 
        commands for now
        """
        
        requires_update = False
        
        ######################################################################
        # Run whatever script commands need to be run.  If we're running a
        # script, always set requires_update
        ######################################################################
        if self.script_running:
            
            requires_update = True
            
            commands = self.script[self.script_idx]
            
            self.script_idx += 1
            
            for command in commands:
                
                if command in self.keyboard_commands:
                    
                    self.keyboard_commands[command]()
                    
            if self.script_idx >= len(self.script):
                
                self.script_running = False
                
                
            
        ######################################################################
        # Run any active keyboard commands
        ######################################################################
        if time.time() - self.last_keypress_time > 0.02:
            
            for curr_key in self.keyboard_active:
            
                if self.keyboard_active[curr_key]:
                    
                    self.keyboard_commands[curr_key]()
                    
                    requires_update = True
                    
        ######################################################################
        # If we did anything that requires a redraw, update the uniform
        # variable with the new camera location for each enabled program,
        # then call self.update() to redraw everything
        ######################################################################
        if requires_update:
            
            for curr_program, draw_type in self.enabled_programs:
                
                curr_program["u_view"] = self.view
            
            self.update()
        
        ######################################################################
        # Lastly, if we're recording, grab the frame after its been drawn
        # from the front buffer and save it to memory
        ######################################################################
        if self.recording:
            
            img = self.read_front_buffer()
        
            #img[:,:,3] = 255
            
            self.recording_frames.append(img)
        
        
        
    def on_resize(self, event):
        
        self.apply_zoom()
        
        
    def mouse_wheel_down(self):
        
        self.point_size_denominator -= 1.0
        
        self.point_size_denominator = max(1.0, self.point_size_denominator)
        
        self.point_size_denominator = min(10.0, self.point_size_denominator)
    
        self.galaxy_point_program['u_size'] = 1.0 / self.point_size_denominator
        
        self.update()
        
    def mouse_wheel_up(self):
        
        self.point_size_denominator += 1.0
        
        self.point_size_denominator = max(1.0, self.point_size_denominator)
        
        self.point_size_denominator = min(10.0, self.point_size_denominator)
    
        self.galaxy_point_program['u_size'] = 1.0 / self.point_size_denominator
        
        self.update()

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
        
        #self.galaxy_point_program.draw('points')
        
        #self.void_sphere_program.draw('triangles')
        
        for curr_program, draw_type in self.enabled_programs:
            
            curr_program.draw(draw_type)
        
        
        

    def apply_zoom(self):
        """
        For a fov angle of 60 degrees and a near plane of 0.01, 
        the size of the near viewing plane is .0115 in height and .0153 in width (on 800x600 screen)
        
        Assuming this won't have any OpenGL programs which don't take the u_projection and
        u_view uniform variables
        """
        
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        
        self.projection = perspective(60.0, self.size[0]/float(self.size[1]), 0.01, 10000.0)
        
        '''
        self.galaxy_point_program['u_projection'] = self.projection
        
        self.galaxy_point_program['u_view'] = self.view
        
        self.void_sphere_program['u_projection'] = self.projection
        
        self.void_sphere_program['u_view'] = self.view
        '''
        
        for curr_program, draw_type in self.enabled_programs:
            
            curr_program["u_projection"] = self.projection
            
            curr_program["u_view"] = self.view
        
        
        
        
        
        
    def run(self):
        app.run()


if __name__ == "__main__":

    from vispy.color import Colormap

    from load_results import load_void_data, load_galaxy_data

    
    voids_tri_x, voids_tri_y, voids_tri_z, voids_norm, voids_id, gal_viz, gal_opp = load_void_data("../data/DR7_triangles.dat","../data/DR7_galviz.dat")
    
    galaxy_data = load_galaxy_data("../data/vollim_dr7_cbp_102709.fits")
    
    print("Voids: ", voids_tri_x.shape, voids_tri_y.shape, voids_tri_z.shape, voids_norm.shape, voids_id.shape)
    
    num_voids = len(np.unique(voids_id))
    
    cm = Colormap(['#880000',
                   '#EEEE00',
                   "#008800",
                   '#EE00EE',
                   '#000088',
                   '#EE00EE'])
    
    void_color_vals = cm.map(np.linspace(0, 1.0, num_voids))
    
    print(void_color_vals.shape)
    
    void_colors = np.empty((num_voids,4), dtype=np.float32)
    
    for idx in range(void_colors.shape[0]):
        
        void_id = idx 
        
        #print(hole_group)
        
        void_colors[idx,:] = void_color_vals[void_id]
            
    
    
    
    print("Galaxies: ", galaxy_data.shape)

    viz = VoidRender(voids_tri_x=voids_tri_x,
                     voids_tri_y=voids_tri_y,
                     voids_tri_z=voids_tri_z,
                     voids_norm=voids_norm,
                     voids_id=voids_id,
                     galaxy_xyz=galaxy_data,
                     galaxy_display_radius=4,
                     gal_viz = gal_viz,
                     gal_opp = gal_opp,
                     #void_hole_color=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                     void_color=void_colors,
                     canvas_size=(1600,1200))
    
    viz.run()
    #app.run()
    
    
    
