#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.optimize

import optrace.tracer.color as color


def xy_point(params):
    mu, sig = params

    xc = np.sum(color.x_tristimulus(wl) * color.gauss(wl, mu, sig))
    yc = np.sum(color.y_tristimulus(wl) * color.gauss(wl, mu, sig))
    zc = np.sum(color.z_tristimulus(wl) * color.gauss(wl, mu, sig))
    bsum = xc + yc + zc

    cord = np.array([xc / bsum, yc / bsum])

    return cord


def xy_point_gauss2(params):
    mu, sig, mu2, sig2, k = params

    xc = np.sum(color.x_tristimulus(wl) * (color.gauss(wl, mu, sig) + k*color.gauss(wl, mu2, sig2)))
    yc = np.sum(color.y_tristimulus(wl) * (color.gauss(wl, mu, sig) + k*color.gauss(wl, mu2, sig2)))
    zc = np.sum(color.z_tristimulus(wl) * (color.gauss(wl, mu, sig) + k*color.gauss(wl, mu2, sig2)))
    bsum = xc + yc + zc

    cord = np.array([xc / bsum, yc / bsum])

    return cord


def cost(params, cord_soll):
    cord_ist = xy_point(params)
    diff = cord_ist - cord_soll

    return np.hypot(diff[0], diff[1])*100


def cost_gauss2(params, cord_soll):
    cord_ist = xy_point_gauss2(params)
    diff = cord_ist - cord_soll

    return np.hypot(diff[0], diff[1])*100


wl = np.linspace(380, 780, 10000)


cord_soll_x = color.SRGB_R_XY
cord_soll_y = color.SRGB_G_XY
cord_soll_z = color.SRGB_B_XY


res_y = scipy.optimize.minimize(cost, x0=[560, 20], args=(cord_soll_y), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 100},
                                bounds=((400, 650), (20, 100)), method='Powell')

res_z = scipy.optimize.minimize(cost_gauss2, x0=[470, 20, 450, 60, 0.2], args=(cord_soll_z), tol=1e-12, options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 300},
                                bounds=((430, 500), (5, 60), (400, 500), (40, 100), (0, 1)), method='Powell')

res_x = scipy.optimize.minimize(cost_gauss2, x0=[640, 30, 420, 80, 0.05], args=(cord_soll_x), tol=1e-12, 
                                options={'ftol':1e-12, 'xtol':1e-12, 'maxiter': 300},
                                bounds=((590, 640), (30, 100), (400, 650), (30, 100), (0, 0.3)), method='Powell')


print("r color")
print(res_x.x)
print(xy_point_gauss2(res_x.x))


print("\ng color")
print(res_y.x)
print(xy_point(res_y.x))

print("\nb color")
print(res_z.x)
print(xy_point_gauss2(res_z.x))


r = color.gauss(wl, *res_x.x[:2]) + res_x.x[4]*color.gauss(wl, *res_x.x[2:4])
g = color.gauss(wl, *res_y.x)
b = color.gauss(wl, *res_z.x[:2]) + res_z.x[4]*color.gauss(wl, *res_z.x[2:4])

# scale such that the maximum is 1
wl_r_max = wl[np.argmax(r)]
norm_r = 1 / (color.gauss(wl_r_max, *res_x.x[:2]) + res_x.x[4]*color.gauss(wl_r_max, *res_x.x[2:4]))
norm_g = 1 / color.gauss(np.array(res_y.x[0]), *res_y.x)
wl_b_max = wl[np.argmax(b)]
norm_b = 1 / (color.gauss(wl_b_max, *res_z.x[:2])+ res_z.x[4]*color.gauss(wl_b_max, *res_z.x[2:4]))


# scale rgb curves
r *= norm_r
g *= norm_g
b *= norm_b

# scale factors for same Y brightness as the sRGB primaries
Y_norm_r = 0.2126729 / np.sum(color.y_tristimulus(wl) * r)
Y_norm_g = 0.7151522 / np.sum(color.y_tristimulus(wl) * g)
Y_norm_b = 0.0721750 / np.sum(color.y_tristimulus(wl) * b)

# normalize the Y_norm scale values such that Y_norm_g = 1
norm2 = Y_norm_g
Y_norm_r /= norm2
Y_norm_g /= norm2
Y_norm_b /= norm2

# plot r, g, b with normalized height
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
fsize=(6,2.6)
plt.figure(figsize=fsize)
plt.plot(wl, r, 'r')
plt.plot(wl, g, 'g')
plt.plot(wl, b, 'b')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
plt.minorticks_on()
plt.xlabel(r"$\lambda$ in nm")
plt.ylabel("arb. unit")
plt.legend(["red", "green", "blue"])
plt.tight_layout()
plt.show(block=False)

# scale rgb curves
r *= Y_norm_r
g *= Y_norm_g
b *= Y_norm_b

# scale factor for same total are under the curve (=total power), normalized such that A_norm_g = 1
A_norm_r = np.sum(r)/np.sum(g)
A_norm_g = 1
A_norm_b = np.sum(b)/np.sum(g)

print("Normalization factors:", norm_r, norm_g, norm_b)
print("Normalization by area: (with A_norm_g = 1)", A_norm_r, A_norm_g, A_norm_b)
print("Normalization by Y stimulus: (with Y_norm_g = 1)", Y_norm_r, Y_norm_g, Y_norm_b)


# plot r, g, b with equal same relative brightness between r, g, b as sRGB primaries
plt.figure(figsize=fsize)
plt.plot(wl, r, 'r')
plt.plot(wl, g, 'g')
plt.plot(wl, b, 'b')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
plt.minorticks_on()
plt.xlabel(r"$\lambda$ in nm")
plt.ylabel("arb. unit")
plt.legend(["red", "green", "blue"])
plt.tight_layout()
plt.show(block=False)


plt.figure(figsize=fsize)
plt.plot(wl, r+g+b, 'r')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor', color='gainsboro', linestyle='--')
plt.minorticks_on()
plt.xlabel(r"$\lambda$ in nm")
plt.ylabel("arb. unit")
plt.tight_layout()
plt.show()

