#!/bin/env python3

from scipy.interpolate import RectBivariateSpline
import scipy.special
import numpy as np

import optrace as ot


def hurb_lens(n:            float, 
              ri:           float, 
              wl:           float, 
              zd:           float, 
              N:            int, 
              N_px:         int, 
              dim_ext_fact: float = 6, 
              use_hurb:     bool = True, 
              hurb_factor:  float = None)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a setup of an aperture directly before an ideal lens using HURB.
    Calculates the PSF at the focal point plane.

    :param n: ambient refractive index
    :param ri: aperture radius in mm
    :param wl: wavelength in nm
    :param zd: detector and focus distance in mm
    :param N: number of rays to trace
    :param N_px: number of detector pixels in each dimension
    :param dim_ext_fact: scaling factor for detector extent
    :param use_hurb: simulate using HURB
    :param hurb_factor: HURB uncertainty factor
    :return: radial distances, simulated data, data from theory
    """
    # see https://en.wikipedia.org/wiki/Airy_disk#Mathematical_formulation
    def rayleigh_curve(x, wl, n, r, z):
        Rnz = 2*np.pi/(wl*1e-9)*n*r/z*x*1e-3
        return (2*scipy.special.j1(Rnz) / Rnz) ** 2

    # make raytracer
    RT = ot.Raytracer(outline=[-15, 15, -15, 15, -6, zd+10], use_hurb=use_hurb, 
                      n0=ot.RefractionIndex("Constant", n))

    if hurb_factor is not None:
        RT.HURB_FACTOR = hurb_factor

    # add ray source
    RS = ot.RaySource(ot.CircularSurface(r=ri), s=[0, 0, 1], pos=[0, 0, -5], 
                      spectrum=ot.LightSpectrum("Monochromatic", wl=wl))
    RT.add(RS)

    # diffracting aperture
    ap_surf = ot.RingSurface(r=ri+1, ri=ri)
    ap = ot.Aperture(ap_surf, pos=[0, 0, -0.001])
    RT.add(ap)

    # add ideal lens
    L = ot.IdealLens(ri+1, 1/zd*1000, pos=[0, 0, 0])
    RT.add(L)

    # add Detector
    # calculate automatic detector width
    dim_ext = 1.22 / (2*np.pi / (wl*1e-9) * n * ri / zd / np.pi) * 1e3 * 6 * dim_ext_fact
    DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
    Det = ot.Detector(DetS, pos=[0, 0, zd])
    RT.add(Det)

    # trace
    RT.trace(N)

    # get irradiance profile along image axis
    img = RT.detector_image()
    imgi = img.get("Irradiance", N_px)
    bins, imgic1 = imgi.profile(x=0)
    bins, imgic2 = imgi.profile(y=0)
    imgic = 0.5*(imgic1[0] + imgic2[0])  # average of x and y profile for less noise
    imgic /= np.max(imgic)

    # radial vector
    r = bins[:-1]+(bins[1]-bins[0])/2

    return r, imgic, rayleigh_curve(r, wl, n, ri, zd)


def hurb_pinhole(n:            float, 
                 ri:           float, 
                 wl:           float, 
                 zd:           float, 
                 N:            int, 
                 N_px:         int, 
                 dim_ext_fact: float = 6, 
                 use_hurb:     bool = True, 
                 hurb_factor:  float = None)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a diffracting pinhole using HURB.
    Calculates the ray profile at a given distance.

    :param n: ambient refractive index
    :param ri: pinhole radius in mm
    :param wl: wavelength in nm
    :param zd: detector distance in mm
    :param N: number of rays to trace
    :param N_px: number of detector pixels in each dimension
    :param dim_ext_fact: scaling factor for detector extent
    :param use_hurb: simulate using HURB
    :param hurb_factor: HURB uncertainty factor
    :return: radial distances, simulated data, data from theory
    """
    # see https://de.wikipedia.org/wiki/Beugungsscheibchen#Beugung_an_einer_Kreisblende
    def rayleigh_curve(x, wl, n, r, z):
        Rnz = 2*np.pi/(wl*1e-9)*n*r/z*x*1e-3
        return (2*scipy.special.j1(Rnz) / Rnz) ** 2

    # make raytracer
    RT = ot.Raytracer(outline=[-15, 15, -15, 15, -6, zd+10], use_hurb=use_hurb, 
                      n0=ot.RefractionIndex("Constant", n))

    if hurb_factor is not None:
        RT.HURB_FACTOR = hurb_factor

    # add ray source
    RS = ot.RaySource(ot.CircularSurface(r=ri), s=[0, 0, 1], pos=[0, 0, -5], 
                      spectrum=ot.LightSpectrum("Monochromatic", wl=wl))
    RT.add(RS)


    # Pinhole
    ap_surf = ot.RingSurface(r=ri+5, ri=ri)
    ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
    RT.add(ap)

    # add Detector
    # detector width
    dim_ext = 1.22 / (2*np.pi / (wl*1e-9) * n * ri / zd / np.pi) * 1e3 * 6 * dim_ext_fact
    DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
    Det = ot.Detector(DetS, pos=[0, 0, zd])
    RT.add(Det)

    # trace
    RT.trace(N)

    # get irradiance profile along image axis
    img = RT.detector_image()
    imgi = img.get("Irradiance", N_px)
    bins, imgic1 = imgi.profile(x=0)
    bins, imgic2 = imgi.profile(y=0)
    imgic = 0.5*(imgic1[0] + imgic2[0])  # average of x and y profile for less noise
    imgic /= np.max(imgic)

    # radial vector
    r = bins[:-1]+(bins[1]-bins[0])/2
    
    return r, imgic, rayleigh_curve(r, wl, n, ri, zd)


def hurb_slit(n:            float, 
              d1:           float, 
              d2:           float, 
              wl:           float, 
              zd:           float, 
              N:            int, 
              N_px:         int, 
              angle:        float = 0,
              dim_ext_fact: float = 6, 
              use_hurb:     bool = True, 
              hurb_factor:  float = None)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a diffracting slit using HURB.
    Calculates the ray profile at a given distance.
    The slit has a width of d1 in x-direction and height d2 in y-direction.
    It is rotated afterwards by 'angle' inside the aperture plane.
    This functions returns the calculated profiles along the rotated d1 and d2 axis.

    :param n: ambient refractive index
    :param d1: slit diameter in x-direction (before rotation) in mm
    :param d2: slit diameter in y-direction (before rotation) in mm
    :param wl: wavelength in nm
    :param zd: detector distance in mm
    :param N: number of rays to trace
    :param N_px: number of detector pixels in each dimension
    :param angle: rotation angle in degrees
    :param dim_ext_fact: scaling factor for detector extent
    :param use_hurb: simulate using HURB
    :param hurb_factor: HURB uncertainty factor
    :return: radial distances along d1 axis, radial distance along d2-axis, simulated data d1-axis, 
             simulated data d2-axis, data from theory d1-axis, data from theory d2-axis
    """

    # see https://en.wikipedia.org/wiki/Diffraction#Single-slit_diffraction
    def slit_curve(x, wl, n, d, z):
        return np.sinc(d*1e-3*n/(wl*1e-9) * x / z)**2
    
    # calculate detector width automatically
    dim_ext = 5 / (min(d1, d2)*1e-3*n/(wl*1e-9) / zd) * dim_ext_fact

    # make raytracer
    RT = ot.Raytracer(outline=[-dim_ext, dim_ext, -dim_ext, dim_ext, -6, zd+10], use_hurb=use_hurb, 
                      n0=ot.RefractionIndex("Constant", n))

    if hurb_factor is not None:
        RT.HURB_FACTOR = hurb_factor

    # add ray source
    rect = ot.RectangularSurface(dim=[d1, d2])
    RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, 0, -5],
                      spectrum=ot.LightSpectrum("Monochromatic", wl=wl))
    RS.rotate(angle)
    RT.add(RS)

    # Slit
    ap_surf = ot.SlitSurface(dim=[d1+2, d2+2], dimi=[d1, d2])
    ap = ot.Aperture(ap_surf, pos=[0, 0, 0])
    ap.rotate(angle)
    RT.add(ap)

    # add Detector
    DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
    Det = ot.Detector(DetS, pos=[0, 0, zd])
    RT.add(Det)

    # trace
    RT.trace(N)

    # get irradiance profile along image axis
    img = RT.detector_image()
    imgi = img.get("Irradiance", N_px)

    # create interpolation object
    x_i = np.linspace(imgi.extent[0], imgi.extent[1], N_px)
    y_i = np.linspace(imgi.extent[2], imgi.extent[3], N_px)
    interpolator = RectBivariateSpline(y_i, x_i, imgi.data, kx=3, ky=3)

    # calculate interpolation axis
    r = np.linspace(img.extent[0], img.extent[1], N_px)
    ang_rad = np.deg2rad(angle)
    rx1 = r * np.cos(ang_rad)
    ry1 = r * np.sin(ang_rad)
    rx2 = r * np.cos(ang_rad+np.pi/2)
    ry2 = r * np.sin(ang_rad+np.pi/2)

    # interpolate and normalize data
    imgic1 = interpolator(ry1, rx1, grid=False)
    imgic2 = interpolator(ry2, rx2, grid=False)
    imgic1 /= np.max(imgic1)
    imgic2 /= np.max(imgic2)

    return r, r, imgic1, imgic2, slit_curve(r, wl, n, d1, zd), slit_curve(r, wl, n, d2, zd)


def hurb_edge(n:            float, 
              wl:           float, 
              zd:           float, 
              N:            int, 
              N_px:         int, 
              dim_ext_fact: float = 6, 
              use_hurb:     bool = True, 
              hurb_factor:  float = None)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a diffracting edge using HURB.
    Calculates the ray profile at a given distance.

    :param n: ambient refractive index
    :param wl: wavelength in nm
    :param zd: detector distance in mm
    :param N: number of rays to trace
    :param N_px: number of detector pixels in each dimension
    :param dim_ext_fact: scaling factor for detector extent
    :param use_hurb: simulate using HURB
    :param hurb_factor: HURB uncertainty factor
    :return: distances along axis orthogonal to edge, simulated data along axis, data from theory along axis
    """
    # see eq 10.132 in https://farside.ph.utexas.edu/teaching/315/Waveshtml/node102.html
    # and eq. 10.99 in  https://farside.ph.utexas.edu/teaching/315/Waveshtml/node99.html 
    def edge_curve(x, wl, n, d, z):
        u_ = np.sqrt(2 * n / (wl*1e-9) / (zd*1e-3)) * x*1e-3
        S, C = scipy.special.fresnel(u_)
        return 0.5*((S+0.5)**2 + (C+0.5)**2)
    
    # calculate detector width
    dim_ext = 0.5*2*dim_ext_fact

    # make raytracer
    RT = ot.Raytracer(outline=[-4*dim_ext, 4*dim_ext, -4*dim_ext, 4*dim_ext, -6, zd+10], use_hurb=use_hurb, 
                      n0=ot.RefractionIndex("Constant", n))

    if hurb_factor is not None:
        RT.HURB_FACTOR = hurb_factor
   
    # ray source
    rect = ot.RectangularSurface(dim=[dim_ext/2, dim_ext/2])
    RS = ot.RaySource(rect, s=[0, 0, 1], pos=[0, dim_ext/4, -1],
                      spectrum=ot.LightSpectrum("Monochromatic", wl=wl))  # offset so edge lies at y=0
    RT.add(RS)

    # large rectangular surface, we only use one edge
    # make sure the width and height are much larger than the used part of the aperture
    ap_surf = ot.SlitSurface(dim=[4*dim_ext, 4*dim_ext], dimi=[4*dim_ext-0.4, 4*dim_ext-0.4])
    ap = ot.Aperture(ap_surf, pos=[0, (4*dim_ext - 0.4)/2, 0])  # offset so edge lies at y=0
    RT.add(ap)

    # add Detector
    DetS = ot.RectangularSurface(dim=[dim_ext, dim_ext])
    Det = ot.Detector(DetS, pos=[0, 0, zd])
    RT.add(Det)

    # trace
    RT.trace(N)
  
    # get irradiance profile along image axis
    img = RT.detector_image()
    imgi = img.get("Irradiance", N_px)
    imgic = np.mean(imgi.data, axis=1)
    imgic /= np.mean(imgic[4*(imgic.shape[0]//5):])

    # radial vector
    r = np.linspace(imgi.extent[2], imgi.extent[3], imgi.shape[0])
    
    return r, imgic, edge_curve(r, wl, n, 5, zd)

