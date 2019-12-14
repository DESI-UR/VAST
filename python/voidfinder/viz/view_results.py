

import numpy as np
#import matplotlib.pyplot as plt

from load_results import load_hole_data, load_galaxy_data




from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate

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
attribute vec3  a_position;
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
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
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


# ------------------------------------------------------------ Canvas class ---
class Canvas(app.Canvas):

    def __init__(self, holes_xyz, holes_radii, galaxy_xyz):
        
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        
        ps = self.pixel_scale
        
        '''
        # Create vertices
        n = 1000000
        
        data = np.zeros(n, [('a_position', np.float32, 3),
                                       ('a_bg_color', np.float32, 4),
                                       ('a_fg_color', np.float32, 4),
                                       ('a_size', np.float32)])
        
        data['a_position'] = 0.45 * np.random.randn(n, 3)
        data['a_bg_color'] = np.random.uniform(0.85, 1.00, (n, 4))
        data['a_fg_color'] = 0, 0, 0, 1
        data['a_size'] = np.random.uniform(5*ps, 10*ps, n)
        '''
        
        
        print("Max xyz: ", holes_xyz.max())
        print("Min xyz: ", holes_xyz.min())
        print("Max radii: ", holes_radii.max())
        print("Min radii: ", holes_radii.min())
        
        num_hole = holes_xyz.shape[0]
        
        num_gal = galaxy_xyz.shape[0]
        
        num_pts = num_hole + num_gal
        
        
        self.hole_data = np.zeros(num_pts, [('a_position', np.float32, 3),
                                       ('a_bg_color', np.float32, 4),
                                       ('a_fg_color', np.float32, 4),
                                       ('a_size', np.float32)])
        
        scale_factor = np.max(np.abs(holes_xyz))
        
        print("Scale factor: ", scale_factor)
        
        #self.position_display_data = np.concatenate((holes_xyz, galaxy_xyz), axis=0)
        
        self.hole_data['a_position'] = np.concatenate((holes_xyz, galaxy_xyz), axis=0)
        
        bg_color_hole = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
        bg_color_gal = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        bg_color_all = np.concatenate((np.tile(bg_color_hole, (num_hole,1)), np.tile(bg_color_gal, (num_gal,1))), axis=0)
        self.hole_data['a_bg_color'] = bg_color_all
        #hole_data['a_bg_color'] = np.random.uniform(0.85, 1.00, (num_pts, 4))
        self.hole_data['a_fg_color'] = 0, 0, 0, 1
        
        sizes_gal = 2.0*np.ones(num_gal)
        
        size_all = np.concatenate((holes_radii, sizes_gal))
        
        self.hole_data['a_size'] = size_all
        #hole_data['a_size'] = np.random.uniform(5*ps, 10*ps, num_pts)
        
        
        #self.mean_location = np.mean(holes_xyz, axis=0)
        
        #print("Mean location: ", mean_location)
        
        
        self.view_location = np.mean(holes_xyz, axis=0)
        self.view_location[2] += 300.0
        self.view_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        self.view_radius = 200.0
        
        
        u_linewidth = 1.0
        u_antialias = 1.0

        self.translate = 5
        self.program = gloo.Program(vert, frag)
        self.view = translate(-self.view_location)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()

        self.program.bind(gloo.VertexBuffer(self.hole_data))
        self.program['u_linewidth'] = u_linewidth
        self.program['u_antialias'] = u_antialias
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_size'] = 5 / self.translate

        self.theta = 0
        self.phi = 0

        gloo.set_state('translucent', clear_color='white')
        
        
        self.mouse_state = 0
        self.last_mouse_pos = [0.0, 0.0]
        

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_timer(self, event):
        
        #self.theta += .5
        #self.phi += .5
        
        self.model = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
        self.program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        
        #print(vars(event))
        #print(dir(event))
        
        #print(event)
        
        #self.translate -= event.delta[1]
        
        translate_factor = 5 * -event.delta[1]
        #self.translate = max(2, self.translate)
        
        #print("Before: ", self.view_location, self.translate)
        
        self.view_location += translate_factor*self.view_vector
        
        self.view = translate(-self.view_location)
        
        #print("After: ", self.view_location)

        self.program['u_view'] = self.view
        
        #self.program['u_size'] = 5 / self.translate
        
        self.update()
        
    def on_mouse_press(self, event):
        #self.print_mouse_event(event, 'Mouse press')
        #print("Press")
        
        if self.mouse_state == 0:
            self.mouse_state = 1
            
            self.last_mouse_pos = event.pos

    def on_mouse_release(self, event):
        #self.print_mouse_event(event, 'Mouse release')
        #print("Release")
        
        if self.mouse_state == 1:
            self.mouse_state = 0

    def on_mouse_move(self, event):
        
        #if (event.pos[0] < self.size[0]*0.5 and event.pos[1] < self.size[1]*0.5):
            
        if self.mouse_state == 1 and event.button == 1:
            
            
            #print(dir(event))
            #print(event.button)
            #print("Pressed Move")
            
            #print(self.last_mouse_pos, event.pos)
            
            x_delta = self.last_mouse_pos[0] - event.pos[0]
            y_delta = self.last_mouse_pos[1] - event.pos[1]
            
            self.last_mouse_pos = event.pos
            
            #print(x_delta, y_delta)
            
            
            self.theta -= y_delta/20.0
            self.phi -= x_delta/20.0
            
            self.model = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
            self.program['u_model'] = self.model
            self.update()
            
        elif self.mouse_state == 1 and event.button == 2:
            
            
            #print(event.button)
            #print("Pressed Move")
            
            #print(self.last_mouse_pos, event.pos)
            
            x_delta = self.last_mouse_pos[0] - event.pos[0]
            y_delta = self.last_mouse_pos[1] - event.pos[1]
            
            self.last_mouse_pos = event.pos
            
            #print(x_delta, y_delta)
            
            #self.last_mouse_pos = event.pos
            
            #self.hole_data['a_position'][:,0] += x_delta
            #self.hole_data['a_position'][:,1] += y_delta
            
            
            self.view_location[0] += x_delta/10.0
            self.view_location[1] -= y_delta/10.0
        
            self.view = translate(-self.view_location)
            
            self.program['u_view'] = self.view
            
            
            #self.program['a_position'] = self.hole_data['a_position']
            
            self.update()
            
            
            
            
            #print(event.pos)
            
            #print(dir(event))
            #print(event.delta, event.pos)
        
        #self.print_mouse_event(event, 'Mouse move')
            
    #def print_mouse_event(self, event, what):
    #    modifiers = ', '.join([key.name for key in event.modifiers])
    #    print('%s - pos: %r, button: %i, modifiers: %s, delta: %r' %
    #          (what, event.pos, event.button, modifiers, event.delta))

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(120.0, self.size[0]/float(self.size[1]), 1.0, 10000.0)
        self.program['u_projection'] = self.projection



if __name__ == "__main__":
    
    holes_xyz, holes_radii, holes_flags = load_hole_data("vollim_dr7_cbp_102709_holes.txt")
    
    #galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
    galaxy_data = load_galaxy_data("dr12r.dat")
    
    print("Holes: ", holes_xyz.shape, holes_radii.shape, holes_flags.shape)
    print("Galaxies: ", galaxy_data.shape)
    
    c = Canvas(holes_xyz, holes_radii, galaxy_data)
    
    app.run()
    
    
    