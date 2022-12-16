#!/usr/bin/env python3

import numpy as np
import optrace.tracer.color as color
import scipy.optimize


def cost_func0(wl, ref):
    wl_ = np.array(wl)
    XYZ = np.array([np.vstack((color.x_observer(wl_),
                               color.y_observer(wl_),
                               color.z_observer(wl_)))])
    XYZs = np.swapaxes(XYZ, 2, 1)
    XYZ2 = color.srgb_linear_to_xyz(color.xyz_to_srgb_linear(XYZs, rendering_intent="Absolute"))
    xy = color.xyz_to_xyY(XYZ2)[0, 0, :2]

    cost = np.sum((xy - ref)**2)
    # print(xy)
    return cost

def cost_func(args, lines):
    # rgb lines have same dominant wavelength as rgb primaries
    # so with srgb conversion with mode "Absolute" we should get the same chromacities as the primaries
    BGR = np.array(lines)
    XYZ = np.array([np.vstack((color.x_observer(BGR) * args,
                               color.y_observer(BGR) * args,
                               color.z_observer(BGR) * args))])
    XYZs = np.array([np.sum(XYZ, axis=2)])
    xy = color.xyz_to_xyY(XYZs)[0, 0, :2]

    cost = np.sum((xy - color.SRGB_W_XY)**2)
    # print(xy)
    return cost

resb = scipy.optimize.minimize(cost_func0, x0=[460], args=(color.SRGB_B_XY), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 100},
                              bounds=((400, 500),), method='Powell')
resg = scipy.optimize.minimize(cost_func0, x0=[540], args=(color.SRGB_G_XY), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 100},
                              bounds=((500, 580),), method='Powell')
resr = scipy.optimize.minimize(cost_func0, x0=[610], args=(color.SRGB_R_XY), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 100},
                              bounds=((580, 650),), method='Powell')


lines = [resb.x[0], resg.x[0], resr.x[0]]
print(lines)

res = scipy.optimize.minimize(cost_func, x0=[0.5, 0.5, 0.5], args=(lines), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 100},
                              bounds=((0.1, 1), (0.1, 1), (0.1, 1)), method='Powell')

print(res)

