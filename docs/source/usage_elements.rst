Defining Elements (Lens, RaySource, Detector, Filter, Aperture, Maker)
------------------------------------------------------------------------


.. testsetup:: *

   import optrace as ot
   import numpy as np


RaySource
_______________________

**Surface/Point/Line Parameter**

.. testcode::

   circ = ot.CircularSurface(r=3)
   RS = ot.RaySource(circ)

This first parameter can be of type ``Point, Line, CircularSurface, RectangularSurface, RingSurface``.


**Position Parameter**

.. testcode::

   RS = ot.RaySource(circ, pos=[0, 1.2, -3.5])

**Power Parameter**

.. testcode::

   RS = ot.RaySource(circ, power=0.5)

**Orientation Parameter**

.. testcode::

   RS = ot.RaySource(circ, orientation="Constant", s=[0.7, 0, 0.7])

.. testcode::

   RS = ot.RaySource(circ, orientation="Constant", s_sph=[20, -30])

.. testcode::

   def or_func(x, y, g=5):
       s = np.column_stack((-x, -y, np.ones_like(x)*g))
       ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
       return s / ab[:, np.newaxis]
   
   RS = ot.RaySource(circ, orientation="Function", or_func=or_func)


**Spectrum Parameter**

.. testcode::

   RS = ot.RaySource(circ, spectrum=ot.presets.light_spectrum.d75)

.. testcode::

   spec = ot.LightSpectrum("Monochromatic", wl=529)
   RS = ot.RaySource(circ, spectrum=spec)


**Divergence Parameter**

.. testcode::

   RS = ot.RaySource(circ, divergence="None", s=[0.7, 0, 0.7])

.. testcode::

   RS = ot.RaySource(circ, divergence="Lambertian", div_angle=10)

.. testcode::

   RS = ot.RaySource(circ, divergence="Isotropic", div_angle=10)

.. testcode::

   RS = ot.RaySource(circ, divergence="Function", div_func=lambda e: np.cos(e)**2, div_angle=10)

.. testcode::

   RS = ot.RaySource(circ, divergence="Function", div_func=lambda e: np.cos(e)**2, div_2d=True, div_axis_angle=20, div_angle=10)


**Image Parameter**


.. testcode::

   rect = ot.RectangularSurface(dim=[2, 3])
   RS = ot.RaySource(rect, image=ot.presets.image.racoon)

.. code-block:: python

   RS = ot.RaySource(rect, image="test_image.png")

**Polarization Parameter**

.. testcode::

   RS = ot.RaySource(circ, polarization="Uniform")

.. testcode::

   RS = ot.RaySource(circ, polarization="x")

.. testcode::

   RS = ot.RaySource(circ, polarization="y")

.. testcode::

   RS = ot.RaySource(circ, polarization="xy")

.. testcode::

   RS = ot.RaySource(circ, polarization="Constant", pol_angle=12)

.. testcode::

   RS = ot.RaySource(circ, polarization="List", pol_angles=[0, 45, 90], pol_probs=[0.5, 0.25, 0.25])

.. testcode::

   RS = ot.RaySource(circ, polarization="Function", pol_func=lambda ang: np.exp(-(ang - 30)**2/10))


Lens
________


**Example**

.. testcode:: 

   sph1 = ot.SphericalSurface(r=3, R=10.2)
   sph2 = ot.SphericalSurface(r=3, R=-20)
   n = ot.RefractionIndex("Sellmeier2", coeff=[1.045, 0.266, 0.206, 0, 0])

   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], de=0.5)


.. testcode::

   n2 = ot.RefractionIndex("Constant", n=1.2)
   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], de=0.5, n2=n2)


**Lens Thickness**

.. testcode::

   L = ot.Lens(sph2, sph1, n=n, pos=[0, 2, 10], d=0.5)

.. testcode::

   L = ot.Lens(sph1, sph2, n=n, pos=[0, 2, 10], d1=0.1, d2=0.6)
   
.. doctest::

   >>> L.d
   0.7

.. doctest::
   
   >>> L.de
   0.022566018848339198

.. doctest::

   >>> L.front.ds
   0.4511539144368477


**Paraxial Properties**

.. doctest::

   >>> tma = L.tma()
   >>> tma.efl
   12.749973064518542

.. doctest::

   >>> tma = L.tma(589.2, n0=ot.RefractionIndex("Constant", n=1.1))
   >>> tma.efl
   17.300045148757384


Ideal Lens
_____________


.. testcode::

   IL = ot.IdealLens(r=5, D=12.5, pos=[0, 0, 9.5])

.. testcode::

   n2 = ot.RefractionIndex("Constant", n=1.25)
   IL = ot.IdealLens(r=4, D=-8.2, pos=[0, 0, 9.5], n2=n2)


Filter
___________


.. testcode::

   spec = ot.TransmissionSpectrum("Rectangle", wl0=400, wl1=500, val=0.5)
   circ = ot.CircularSurface(r=5)
   F = ot.Filter(circ, pos=[0, 0, 23.93], spectrum=spec)


.. doctest::

   >>> F.get_color()
   (2.359115924484492e-07, 0.2705811859857049, 0.9999999999999999, 0.9838657805329205)

.. doctest::

   >>> wl = np.array([380, 400, 550])
   >>> F(wl)
   array([0. , 0.5, 0. ])


Aperture
________________

.. testcode::

   ring = ot.RingSurface(ri=0.05, r=5)
   AP = ot.Aperture(ring, pos=[0, 2, 10.1])

Detector
__________________

.. testcode::

   rect = ot.RectangularSurface(dim=[1.5, 2.3])
   Det = ot.Detector(rect, pos=[0, 0, 15.2])

Marker
_________________

.. testcode::

   M = ot.Marker("Text132", pos=[0.5, 9.1, 0.5])

.. testcode::

   M = ot.Marker("Text132", pos=[0.5, 9.1, 0.5], text_factor=2.3, marker_factor=0.5)

.. testcode::

   M = ot.Marker("Text132", pos=[0.5, 9.1, 0.5], label_only=True)

