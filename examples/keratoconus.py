#!/usr/bin/env python3

import numpy as np

import optrace as ot
import optrace.plots as otp


# options:
g = 0.6e3  # object distance
G_alpha = 4  # angle of object in view
P = 3  # pupil diameter
image = ot.presets.image.ETDRS_chart_inverted  # test image
position = "far"  # keratoconus cone position (see "positions" dictionary below)
cases = [0, 4, 7, 14]  # cases to simulate (see "gauss_params" list below)
delta_A = 0 # 1.5  # relative eye adaption / refractive error from correct focus

# Table 1 of Tan et. al (2008) (https://jov.arvojournals.org/article.aspx?articleid=2158188)
#   h0     sigma_x  sigma_y  Number       Category 
gauss_param =\
[[0.0000,  0.0001,  0.0001],  #  0    Healthy: V = 0.00 mm^3
 [0.0051,  0.4183,  0.4729],  #  1    Mild: V < 0.02 mm^3
 [0.0087,  0.4348,  0.5718],  #  2  
 [0.0090,  0.5170,  0.4960],  #  3  
 [0.0101,  0.7323,  0.6944],  #  4    Moderate: V 0.02–0.1 mm^3 
 [0.0118,  0.6581,  0.7755],  #  5  
 [0.0156,  0.6417,  0.6008],  #  6  
 [0.0200,  0.8000,  0.8000],  #  7  
 [0.0246,  1.1821,  0.8553],  #  8    Advanced: V 0.1–0.4 mm^3  
 [0.0269,  0.9700,  0.8823],  #  9 
 [0.0296,  1.1606,  0.8822],  # 10  
 [0.0400,  1.2000,  1.2000],  # 11  
 [0.0410,  1.7380,  1.0590],  # 12    Severe: V > 0.4 mm^3 
 [0.0507,  1.7013,  1.0280],  # 13  
 [0.0541,  1.7629,  1.0309]]  # 14

# position dictionary
# Figure 1 of Tan et. al (2008) (https://jov.arvojournals.org/article.aspx?articleid=2158188)
positions = {"axis": [0., 0.], "average": [0.4, -0.9], "far": [1.1, -1.4]}

# ray settings
N_rays = 3e5

# resulting properties
A = 1/g*1000  + delta_A # adaption in dpt for given g
G = g*np.tan(G_alpha/180*np.pi) # half object size
OL = max(G, 8)  # half of x, y, outline size
sr_angle = np.arctan(1.4*P/2/g)/np.pi*180  # ray divergence angle
G_size = g*np.tan(G_alpha/180*np.pi)  # object size from angle

# create raytracer
RT = ot.Raytracer(outline=[-OL, OL, -OL, OL, -g, 28], no_pol=False)

# Point Source
RS = ot.RaySource(ot.Point(), divergence="Lambertian", div_angle=sr_angle, pos=[0, 0, -g])
RT.add(RS)

# load Eye model
geom = ot.presets.geometry.arizona_eye(adaptation=A, pupil=P)
RT.add(geom)

# add rectangular detector
DetS = ot.RectangularSurface([4, 4])
Det = ot.Detector(DetS, pos=RT.detectors[0].pos, desc="Retina")
RT.add(Det)


# new anterior cornea function
def cornea_ant_func(x, y, cornea_front, gauss_param, position):

    base = cornea_front._values(x, y)
    h, sx, sy = gauss_param
    x0, y0 = position

    return base - h*np.exp(-(x-x0)**2/2/sx**2 - (y-y0)**2/2/sy**2)


# save old cornea
old_cornea = RT.lenses[0]
cornea = old_cornea

for num in cases:
   
    # remove old cornea
    RT.remove(cornea)

    # define new cornea
    func_args=dict(cornea_front=old_cornea.front, gauss_param=gauss_param[num], position=positions[position])
    cfront = ot.FunctionSurface2D(func=cornea_ant_func, func_args=func_args, r=old_cornea.front.r)
    func_args=dict(cornea_front=old_cornea.back, gauss_param=gauss_param[num], position=positions[position])
    cback = ot.FunctionSurface2D(func=cornea_ant_func, func_args=func_args, r=old_cornea.back.r)
    # cornea = ot.Lens(cfront, cback, d1=0, d2=0.55, pos=[0, 0, 0], n=old_cornea.n, n2=old_cornea.n2)
    cornea = ot.Lens(cfront, old_cornea.back, d1=0, d2=0.55, pos=[0, 0, 0], n=old_cornea.n, n2=old_cornea.n2)
    RT.add(cornea)
    
    # render PSF
    det_im = RT.iterative_render(N_rays, detector_index=1, limit=4)

    psf = det_im[0]
    img = image([2*G_size, 2*G_size])

    # calculate image magnification
    m = ot.presets.geometry.arizona_eye().tma().image_magnification(RS.pos[2])

    # convolve with PSF
    img_conv = ot.convolve(img, psf, m=m, slice_=True)
    
    # show image
    otp.image_plot(img_conv, flip=True)

# make plots blocking
otp.block()
