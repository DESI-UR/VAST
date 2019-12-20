

import numpy as np

from load_results import load_hole_data, load_galaxy_data

from unionize import union_vertex_selection

from vispy import gloo

from vispy import app

from vispy.util.transforms import perspective, translate, rotate

from vispy.util.quaternion import Quaternion

from vispy.visuals.transforms import STTransform

from vispy.color import Color

from vispy.geometry import create_box, create_sphere

from vispy import scene

from sklearn import neighbors

import time

vert = """
#version 120
// Uniforms
// ------------------------------------
uniform mat4 u_model;
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
    gl_Position = u_projection * u_view * u_model * a_position;
    
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

#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec4 position;
//attribute vec2 texcoord;
//attribute vec3 normal;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    v_color = color;
    gl_Position = u_projection * u_view * u_model * position;
}
"""

frag_sphere = """

#version 330 core

varying vec4 v_color;

void main()
{

    gl_FragColor = v_color;
    //gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

"""


#Cube method of making a sphere
def _cube(rows, cols, depth, radius):
    # vertices and faces of tessellated cube
    verts, faces, _ = create_box(1, 1, 1, cols, rows, depth)
    verts = verts['position']

    # make each vertex to lie on the sphere
    lengths = np.sqrt((verts*verts).sum(axis=1))
    verts /= lengths[:, np.newaxis]/radius
    #return MeshData(vertices=verts, faces=faces)
    return verts

def _ico(radius, subdivisions):
    # golden ratio
    t = (1.0 + np.sqrt(5.0))/2.0

    # vertices of a icosahedron
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

    # faces of the icosahedron
    faces = [(0, 11, 5),
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

    def midpoint(v1, v2):
        return ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2, (v1[2]+v2[2])/2)

    # subdivision
    for _ in range(subdivisions):
        
        for idx in range(len(faces)):
            i, j, k = faces[idx]
            a, b, c = verts[i], verts[j], verts[k]
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            verts += [ab, bc, ca]
            ij, jk, ki = len(verts)-3, len(verts)-2, len(verts)-1
            faces.append([i, ij, ki])
            faces.append([ij, j, jk])
            faces.append([ki, jk, k])
            faces[idx] = [jk, ki, ij]
    verts = np.array(verts)
    faces = np.array(faces)

    # make each vertex to lie on the sphere
    lengths = np.sqrt((verts*verts).sum(axis=1))
    verts /= lengths[:, np.newaxis]/radius
    return verts


class Triangle(object):
    
    def __init__(self, pt1, pt2, pt3):
        
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3

def _ico2(radius, subdivisions):
    # golden ratio
    t = (1.0 + np.sqrt(5.0))/2.0

    # vertices of a icosahedron
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
    # Subdivide each triangle - method 1
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
    # Subdivide each triangle - method 2
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
    ############################################################
    
    array_data = []
    
    for curr_tri in face_list:
        
        array_data.append(curr_tri.pt1.reshape(1,3))
        array_data.append(curr_tri.pt2.reshape(1,3))
        array_data.append(curr_tri.pt3.reshape(1,3))
        
    out_vertices = np.concatenate(array_data, axis=0)
    
    ############################################################
    # Scale by radius
    ############################################################
    
    out_vertices *= radius
    
    return out_vertices








# ------------------------------------------------------------ Canvas class ---
class Canvas(app.Canvas):
#class Canvas(scene.SceneCanvas):

    def __init__(self,
                 holes_xyz, 
                 holes_radii, 
                 galaxy_xyz,
                 unionize_holes=True,
                 galaxy_display_radius=2.0,
                 canvas_size=(800,600)):
        '''
        Main class for initializing the visualization.
        
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
            
        canvas_size : 2-tuple
            (width, height) in pixels for the output visualization
        '''
        
        
        app.Canvas.__init__(self, 
                            keys='interactive', 
                            size=canvas_size)
        '''
        scene.SceneCanvas.__init__(self,
                                   keys='interactive',
                                   size=canvas_size)
        
        
        self.unfreeze()
        '''
        #print(self.app)
        #print(self.canvas)
        
        self.title = "VoidFinder Results"
        
        ps = self.pixel_scale
        
        self.holes_xyz = holes_xyz
        self.holes_radii = holes_radii
        
        ######################################################################
        # Set up some numpy arrays to work with the vispy/OpenGL vertex 
        # shaders and fragment shaders.  The names (string variables) used
        # in the below code match up to names used in the vertex and
        # fragment shader code strings used above
        ######################################################################
        self.num_hole = holes_xyz.shape[0]
        
        self.num_gal = galaxy_xyz.shape[0]
        
        self.num_pts = self.num_hole + self.num_gal
        
        #self.vertex_data = np.zeros(self.num_pts, [('a_position', np.float32, 4),
        #                                           ('a_bg_color', np.float32, 4),
        #                                           ('a_fg_color', np.float32, 4),
        #                                           ('a_size', np.float32)])
        
        
        self.vertex_data = np.zeros(self.num_gal, [('a_position', np.float32, 4),
                                                   ('a_bg_color', np.float32, 4),
                                                   ('a_fg_color', np.float32, 4),
                                                   ('a_size', np.float32)])
        
        
        
        ######################################################################
        # Concatenate the array of hole locations and galaxy locations
        # into a single array for the vertex and fragment shaders
        # Extract the maximum component from all the data to use as a
        # scale factor to convert from raw xyz space to a (-1.0, 1.0) based
        # OpenGL camera viewing space
        ######################################################################
        #self.all_xyz_coords = np.concatenate((holes_xyz, galaxy_xyz), axis=0)
        
        self.all_xyz_coords = galaxy_xyz
        
        self.scale_factor = np.max(np.abs(self.all_xyz_coords))
        
        ws = np.ones((self.all_xyz_coords.shape[0], 1), dtype=np.float32)
        
        #ws[:,0] = np.sum(self.all_xyz_coords*self.all_xyz_coords, axis=1)/(self.scale_factor)
        
        self.all_xyzw_coords = np.concatenate((self.all_xyz_coords, ws), axis=1)
        
        self.vertex_data['a_position'] = self.all_xyzw_coords
        
        self.kdtree = neighbors.KDTree(self.all_xyz_coords)
        
        #print(self.all_xyz_coords[0:10])
        
        ######################################################################
        # Create the color arrays in RGBA format for the holes and galaxies
        # I believe bg color is 'background' and fg is 'foreground' color
        ######################################################################
        holes_color = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
        
        gal_color = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        
        #bg_color_all = np.concatenate((np.tile(holes_color, (self.num_hole,1)), np.tile(gal_color, (self.num_gal,1))), axis=0)
        
        bg_color_all = np.tile(gal_color, (self.num_gal,1))
        
        
        self.vertex_data['a_bg_color'] = bg_color_all
        
        self.vertex_data['a_fg_color'] = (0.0, 0.0, 0.0, 1.0)
        
        #print(self.vertex_data['a_fg_color'])
        
        ######################################################################
        # Since the data is currently being displayed as disks instead of
        # Spheres, use a fixed radius of 2.0 for galaxies for now since they
        # are small compared to void holes
        #
        # Then concatenate them with the hole radii so they match up
        # with the concatenated hole and galaxy locations
        #
        ######################################################################
        sizes_gal = 2.0*np.ones(self.num_gal)
        
        #self.size_all = np.concatenate((holes_radii, sizes_gal))
        self.size_all = sizes_gal
        
        self.vertex_data['a_size'] = self.size_all
        
        ######################################################################
        # Set up the initial camera location.  Camera is controlled
        # by setting "self.program['u_view'] = self.view" which becomes
        # part of:
        # "gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);"
        # 
        # Currently:
        #    z axis - forward backward
        #    y axis - up/down
        #    x axis - left/right
        #
        #
        # self.view_axis_vector is the xyz vector (plus w dimension) encoding
        # the current camera direction
        #
        # self.view_orientation is:
        #
        #  [x_x, x_y, x_z, 0]  x projection row vector
        #  [y_x, y_y, y_z, 0]  y projection row vector
        #  [z_x, z_y, z_z, 0]  z projection row vector
        #  
        #
        #
        ######################################################################
        self.view_location = np.mean(holes_xyz, axis=0)
        
        self.view_location[2] += 300.0
        
        #alpha, x, y, z
        #self.orientation_quaternion = Quaternion(0.0, 0.0, 1.0, 0.0)
        
        #self.view_orientation = np.eye(3, dtype=np.float32)
        self.orientation_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.orientation_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.orientation_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        
        self.translation_sensitivity = 1.0
        
        self.rotation_sensitivity = 1.0
        
        self.view = np.eye(4, dtype=np.float32)
        
        self.view[3,0:3] = -self.view_location
        
        

        ######################################################################
        # Set up the OpenGL program stuff
        ######################################################################
        u_linewidth = 1.0
        
        u_antialias = 0.1
        
        self.program = gloo.Program(vert, frag)
        
        self.model = np.eye(4, dtype=np.float32)
        
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()
        
        
        self.vertex_buffer = gloo.VertexBuffer(self.vertex_data)
        
        self.program.bind(self.vertex_buffer)
        
        self.program['u_linewidth'] = u_linewidth
        self.program['u_antialias'] = u_antialias
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_size'] = 1.0

        self.theta = 0
        self.phi = 0

        ######################################################################
        # Dunno what this does
        ######################################################################
        
        gloo.set_state('translucent', clear_color='white')
        
        ######################################################################
        # Set up user input information
        ######################################################################
        
        self.translate = 1.0
        
        self.last_keypress_time = time.time()
        
        self.mouse_state = 0
        
        self.last_mouse_pos = [0.0, 0.0]
        
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
                                  #'u' : self.press_u,
                                  'l' : self.press_l,
                                  #'o' : self.press_o,
                                  'j' : self.press_j,
                                  'q' : self.press_q,
                                  'e' : self.press_e}
        
        self.press_once_commands = {" " : self.press_spacebar,
                                    "p" : self.press_p}
        
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
        
        ######################################################################
        # Set up a callback to a timer function
        ######################################################################
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)#, app=self.app)
        #self.timer = app.Timer()
        #self.timer.connect(self.on_timer)
        #self.timer.start(interval=1.0/60.0)
        
        
        #self.test_timer = self.app.Timer(1.0, connect=self.on_timer2, start=True)
        
        
        
        ######################################################################
        # Create triangles at origin
        #
        #test_verts["position"] = np.array([[-20.0, 0.0, 0.0, 1.0],
        #                                   [20.0, 0.0, 0.0, 1.0],
        #                                   [0.0, 20.0, 0.0, 1.0],
        #                                   [40.0, 0.0, 0.0, 1.0],
        #                                   [80.0, 0.0, 0.0, 1.0],
        #                                   [60.0, 20.0, 0.0, 1.0]], dtype=np.float32)
        #
        #test_verts["color"] = np.array([[1.0, 0.0, 0.0, 1.0],
        #                                [0.0, 1.0, 0.0, 1.0],
        #                                [0.0, 0.0, 1.0, 1.0],
        #                                [1.0, 0.0, 0.0, 1.0],
        #                                [0.0, 1.0, 0.0, 1.0],
        #                                [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        ######################################################################
        
        
        
        ######################################################################
        # Create a unit sphere composed of triangles for copying
        ######################################################################
        
        
        testing = False
        
        if testing:
            
            self.holes_xyz = self.holes_xyz[0:100]
            self.holes_radii = self.holes_radii[0:100]
            
            
        
        self.num_spheres = self.holes_xyz.shape[0]
            
        
        self.unit_sphere = self.create_sphere(1.0, 3)
        
        #print(self.unit_sphere[0:10])
        
        #print(np.sum(self.unit_sphere*self.unit_sphere, axis=1)[0:10])
        
        self.vert_per_sphere = self.unit_sphere.shape[0]
        
        num_sphere_verts = self.vert_per_sphere*self.num_spheres
        
        self.sphere_coord_data = np.ones((num_sphere_verts,4), dtype=np.float32)
        
        #print(self.sphere_coord_data.shape)
        
        for idx, (hole_xyz, hole_radius) in enumerate(zip(self.holes_xyz, self.holes_radii)):
            
            #if idx >= self.num_spheres:
            #    break
            
            
            #scaled_sphere = self.unit_sphere * hole_radius
            
            #print(np.mean(np.sqrt(np.sum(scaled_sphere*scaled_sphere,axis=1))), hole_radius)
            
            curr_sphere = (self.unit_sphere * hole_radius) + hole_xyz
            
            #print(curr_sphere.shape)
            
            start_idx = idx*self.vert_per_sphere
            end_idx = (idx+1)*self.vert_per_sphere
            
            #print(start_idx, end_idx)
            
            self.sphere_coord_data[start_idx:end_idx, 0:3] = curr_sphere
            
        if unionize_holes:
            
            print("Pre-union vertices: ", self.sphere_coord_data.shape[0])
            
            self.unionize_hole_data()
            
            num_sphere_verts = self.sphere_coord_data.shape[0]
            
            print("Post-union vertices: ", num_sphere_verts)
        
        curr_color = np.array([[0.0, 0.0, 1.0, 0.5]], dtype=np.float32)
        
        out_color = np.tile(curr_color, (self.sphere_coord_data.shape[0], 1))
        
        ######################################################################
        #
        ######################################################################
        
        self.sphere_vertex_buffer = np.zeros(num_sphere_verts, [('position', np.float32, 4),
                                                                ('color', np.float32, 4)])
        
        
        ######################################################################
        # Set up the sphere-drawing program
        ######################################################################
        self.sphere_vertex_buffer["position"] = self.sphere_coord_data
        
        self.sphere_vertex_buffer["color"] = out_color
        
        self.program_sphere = gloo.Program(vert_sphere, frag_sphere)
        
        self.apply_zoom_sphere()
        
        curr_vertex_buffer = gloo.VertexBuffer(self.sphere_vertex_buffer)
        
        self.program_sphere.bind(curr_vertex_buffer)
        
        self.program_sphere['u_model'] = self.model
        
        self.program_sphere['u_view'] = self.view
        
        ######################################################################
        # 
        ######################################################################
        self.show()
        
        
    def create_sphere(self, radius, subdivisions=2):
        
        sphere_vertices = _ico2(radius, subdivisions)
        
        return sphere_vertices
    
    def unionize_hole_data(self):
        
        valid_vertex_idx = np.ones(self.sphere_coord_data.shape[0], dtype=np.uint8)
        
        hole_kdtree = neighbors.KDTree(self.holes_xyz)
        
        neighbor_index = hole_kdtree.query_radius(self.holes_xyz, self.holes_radii)
        
        #print(neighbor_index[0].shape)
        
        #print(neighbor_index.dtype, neighbor_index.shape)
        #print(self.holes_xyz.dtype) #do astype float32
        #print(self.holes_radii.dtype) #do astype float32
        '''
        for curr_idx, (close_idx, hole_xyz, hole_radius) in enumerate(zip(neighbor_index, 
                                                                          self.holes_xyz, 
                                                                          self.holes_radii)):
            
            if curr_idx % 100 == 0:
                print("Unionizing: ", curr_idx)
            
            for neighbor_idx in close_idx:
                
                if neighbor_idx == curr_idx:
                    continue
                
                
                start_idx = neighbor_idx*self.vert_per_sphere
                
                end_idx = (neighbor_idx+1)*self.vert_per_sphere
                
                
                vert_dist_components = self.sphere_coord_data[start_idx:end_idx, 0:3] - hole_xyz
                
                vert_dists = np.sum(vert_dist_components*vert_dist_components, axis=1)
                
                valid_verts = vert_dists >= hole_radius*hole_radius
                
                for offset_idx in range(self.vert_per_sphere//3): #integer division by 3 since every triangle has 3 verts
                    
                    off_start = 3*offset_idx
                    
                    off_end = 3*(offset_idx+1)
                    
                    valid_verts[off_start:off_end] = np.any(valid_verts[off_start:off_end])
                    
                valid_vertex_idx[start_idx:end_idx] = np.logical_and(valid_verts, valid_vertex_idx[start_idx:end_idx])
        '''
        
        #print("Valid pre: ", np.count_nonzero(valid_vertex_idx))
        #print(valid_vertex_idx[0:10])
        #print(np.where(valid_vertex_idx)[0][0:10])
        
        union_vertex_selection(neighbor_index,
                               valid_vertex_idx,
                               self.holes_xyz.astype(np.float32),
                               self.holes_radii.astype(np.float32),
                               self.sphere_coord_data,
                               self.vert_per_sphere)
        
        #print("Valid post: ", np.count_nonzero(valid_vertex_idx))
        #print(valid_vertex_idx[0:10])
        #print(np.where(valid_vertex_idx)[0][0:10])
        
        
        #print(self.sphere_coord_data[valid_vertex_idx].shape)
        
        #needed to use numpy.where maybe due to uint8 type on valid_vertex_idx?
        self.sphere_coord_data = self.sphere_coord_data[np.where(valid_vertex_idx)[0]]
        

    def press_spacebar(self):
        """
        Keeping this function because it shows how to convert from the view matrix
        position to real coordinate space position
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
        
            
    def translate_camera(self, idx, plus_minus):
        
        self.view[3,idx] += plus_minus*self.translation_sensitivity
        
        self.program['u_view'] = self.view
        
        self.program_sphere["u_view"] = self.view
        
        self.update()
        
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
        
        self.program['u_view'] = self.view
        
        self.program_sphere["u_view"] = self.view
        
        self.update()
        
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
        
        self.translation_sensitivity *= 1.2
        
    def press_x(self):
        
        self.translation_sensitivity /= 1.2
        
    def press_c(self):
        
        self.rotation_sensitivity *= 1.2  
        
    def press_v(self):
        
        self.rotation_sensitivity /= 1.2
        
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
        
        curr_camera_location = self.view[3,0:3].reshape(1,3)
        
        curr_rot_matrix = self.view[0:3,0:3]
        
        val = np.matmul(curr_rot_matrix, curr_camera_location.T)
        
        norm = np.sqrt(np.sum(curr_camera_location*curr_camera_location))
        
        #print(curr_camera_location)
        
        print(self.view)
        print(val.T)
        print(norm)
        
        
        

    def on_key_press(self, event):
        """
        Activated when user pressed a key on the keyboard
        """
        if event.text in self.keyboard_commands:
            
            self.keyboard_active[event.text] = 1
            
        elif event.text in self.press_once_commands:
            
            self.press_once_commands[event.text]()
            
        for curr_key in self.keyboard_active:
            
            if self.keyboard_active[curr_key]:
    
                self.keyboard_commands[curr_key]()
            
            
    def on_key_release(self, event):
        
        if event.text in self.keyboard_commands:
            
            #print("Release: ", event.text)
            
            self.keyboard_active[event.text] = 0
    

    def on_timer(self, event):
        
        #print("I am called!")
        
        if time.time() - self.last_keypress_time > 0.02:
            
            for curr_key in self.keyboard_active:
            
                if self.keyboard_active[curr_key]:
                    
                    #print("Running command: ", curr_key)
        
                    self.keyboard_commands[curr_key]()
        
    def on_resize(self, event):
        
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        
        pass
        '''
        #print(vars(event))
        #print(dir(event))
        
        #print(event)
        
        #
        
        translate_factor = 5 * -event.delta[1]
        #self.translate = max(2, self.translate)
        
        #print("Before: ", self.view_location, self.translate)
        
        self.view_location += translate_factor*self.view_vector
        
        self.view = translate(-self.view_location)
        
        #print("After: ", self.view_location)

        self.program['u_view'] = self.view
        '''
    
        self.translate -= event.delta[1]
        
        #print(self.translate)
        
        self.translate = max(1.0, self.translate)
    
        self.program['u_size'] = 1.0 / self.translate
        
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
            '''
            x_delta = self.last_mouse_pos[0] - event.pos[0]
            
            y_delta = self.last_mouse_pos[1] - event.pos[1]
            
            self.last_mouse_pos = event.pos
            
            self.theta -= y_delta/20.0
            
            self.phi -= x_delta/20.0
            
            self.model = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
            
            self.program['u_model'] = self.model
            
            self.update()
            '''
            pass
            
        elif self.mouse_state == 1 and event.button == 2:
            
            pass
            '''
            x_delta = self.last_mouse_pos[0] - event.pos[0]
            
            y_delta = self.last_mouse_pos[1] - event.pos[1]
            
            self.last_mouse_pos = event.pos
            
            #self.view_location[0] += x_delta/10.0
            
            #self.view_location[1] -= y_delta/10.0
        
            self.view = translate(-self.view_location)
            
            self.program['u_view'] = self.view
            
            self.update()
            '''
            
            

    def on_draw(self, event):
        
        gloo.clear((1, 1, 1, 1))
        
        self.program.draw('points')
        
        self.program_sphere.draw('triangles')

    def apply_zoom(self):
        """
        For a fov angle of 60 degrees and a near plane of 0.01, 
        the size of the near viewing plane is .0115 in height and .0153 in width (on 800x600 screen)
        """
        
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        
        self.projection = perspective(60.0, self.size[0]/float(self.size[1]), 0.01, 10000.0)
        
        self.program['u_projection'] = self.projection
        
    def apply_zoom_sphere(self):
        """
        For a fov angle of 60 degrees and a near plane of 0.01, 
        the size of the near viewing plane is .0115 in height and .0153 in width (on 800x600 screen)
        """
        
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        
        self.projection = perspective(60.0, self.size[0]/float(self.size[1]), 0.01, 10000.0)
        
        self.program_sphere['u_projection'] = self.projection
        
        



if __name__ == "__main__":
    
    holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")
    
    galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
    #galaxy_data = load_galaxy_data("dr12r.dat")
    
    print("Holes: ", holes_xyz.shape, holes_radii.shape, holes_flags.shape)
    
    print("Galaxies: ", galaxy_data.shape)
    
    c = Canvas(holes_xyz, 
               holes_radii, 
               galaxy_data)
    
    app.run()
    
    
    