.. _usage_tma:

Paraxial Analysis
-------------------------------

.. role:: python(code)
  :language: python
  :class: highlight

.. |TMA_link| replace:: :class:`TMA <optrace.tracer.transfer_matrix_analysis.TMA>` 

.. testsetup:: *

   import numpy as np
   import optrace as ot

   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

Overview
______________

A ray transfer matrix analysis object |TMA_link| allows for the analysis of the paraxial properties of a lens or lens setups. 
It is computed for the current state of the geometry and stores the properties for a specific wavelength.
These for instance include focal lengths, nodal points, optical powers and more.

Defining the Geometry for the Analysis
__________________________________________

**Group/Raytracer**

From a :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` with a specific geometry the |TMA_link| object is calculated with the member function :meth:`tma() <optrace.tracer.geometry.group.Group.tma>`.

.. testcode::

   tma = RT.tma()

The function optionally takes a wavelength in nanometers as argument, for which the properties are calculated:

.. testcode::

   tma = RT.tma(780)

For a :class:`Group <optrace.tracer.geometry.group.Group>` object this works in the same way:

.. testcode::

   G = ot.presets.geometry.arizona_eye()
   tma = G.tma()


**Single Lens**

We can also create the analysis object for a single :class:`Lens <optrace.tracer.geometry.lens.Lens>`:

.. testcode::

   front = ot.CircularSurface(r=5)
   back = ot.SphericalSurface(r=5, R=-25)
   n = ot.presets.refraction_index.K5
   L = ot.Lens(front, back, n=n, d=0.5, pos=[0, 0, 10])

   tma = L.tma()

While a single Lens defines the subsequent ambient medium, it has no knowledge on the preceding medium.
Normally it will be assigned by either the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` or the previous lens.
The same is the case for the medium :python:`n0` of the raytracer, which defines all undefined :python:`Lens.n2` media.
To define it for the TMA, we can provide a :python:`n0` parameter.
Otherwise it defaults to the vacuum properties.

.. testcode::

   n0 = ot.RefractionIndex("Constant", n=1.1)
   tma = L.tma(n0=n0)

**Multiple Lenses**

Without a specific geometry, we can also create the |TMA_link| object by providing a list of lenses.

.. testcode::

   back2 = ot.SphericalSurface(r=5, R=-25)
   front2 = ot.CircularSurface(r=5)
   n2 = ot.presets.refraction_index.F2
   L2 = ot.Lens(front, back, n=n2, de=0.5, pos=[0, 0, 16])

   Ls = [L, L2]
   tma = ot.TMA(Ls)

As for the lens, the previous ambient medium (or the medium for all undefined :python:`Lens.n2`) can be provided with the :python:`n0` parameter.

.. testcode::

   tma = ot.TMA(Ls, n0=n0)


Paraxial Properties
__________________________________________

The following table provides an overview of supported TMA properties.
Details on their meaning and calculation are documented in :numref:`ray_matrix_analysis` and more information on the different definitions for focal lengths and powers in :numref:`ray_power_def`.

.. list-table:: Properties of a |TMA_link| object
   :widths: 75 60 40 200
   :header-rows: 1
   :align: center

   * - Variable
     - Type
     - Unit
     - Meaning

   * - :attr:`n1 <optrace.tracer.transfer_matrix_analysis.TMA.n1>`
     - :python:`float`
     - ``-``
     - refractive index value before the lens setup
   
   * - :attr:`n2 <optrace.tracer.transfer_matrix_analysis.TMA.n2>`
     - :python:`float`
     - ``-``
     - refractive index value after the lens setup
   
   * - :attr:`vertex_points <optrace.tracer.transfer_matrix_analysis.TMA.vertex_points>`
     - :python:`tuple[float, float]`
     - mm
     - front and back position of vertices of the system 
   
   * - :attr:`d <optrace.tracer.transfer_matrix_analysis.TMA.d>`
     - :python:`float`
     - mm
     - thickness, distance between vertex points
   
   * - :attr:`abcd <optrace.tracer.transfer_matrix_analysis.TMA.abcd>`
     - :class:`numpy.ndarray`, shape (2, 2)
     - ``-``
     - ABCD matrix

   * - :attr:`principal_points <optrace.tracer.transfer_matrix_analysis.TMA.principal_points>`
     - :python:`tuple[float, float]`
     - mm
     - principal points (z-positions)

   * - :attr:`nodal_points <optrace.tracer.transfer_matrix_analysis.TMA.nodal_points>`
     - :python:`tuple[float, float]`
     - mm
     - nodal points (z-positions)
   
   * - :attr:`optical_center <optrace.tracer.transfer_matrix_analysis.TMA.optical_center>`
     - :python:`float`
     - mm
     - optical center (z-position)
   
   * - :attr:`focal_points <optrace.tracer.transfer_matrix_analysis.TMA.focal_points>`
     - :python:`tuple[float, float]`
     - mm
     - focal points (z-positions)
   
   * - :attr:`focal_lengths <optrace.tracer.transfer_matrix_analysis.TMA.focal_lengths>`
     - :python:`tuple[float, float]`
     - mm
     - focal lengths
   
   * - :attr:`focal_lengths_n <optrace.tracer.transfer_matrix_analysis.TMA.focal_lengths_n>`
     - :python:`tuple[float, float]`
     - mm
     - focal lengths, scaled with refractive index

   * - :attr:`powers <optrace.tracer.transfer_matrix_analysis.TMA.powers>`
     - :python:`tuple[float, float]`
     - dpt
     - optical powers of the system
   
   * - :attr:`powers_n <optrace.tracer.transfer_matrix_analysis.TMA.powers_n>`
     - :python:`tuple[float, float]`
     - dpt
     - optical powers, scaled with the refractive index
   
   * - :attr:`efl <optrace.tracer.transfer_matrix_analysis.TMA.efl>`
     - :python:`float`
     - mm
     - effective focal length of the system

   * - :attr:`efl_n <optrace.tracer.transfer_matrix_analysis.TMA.efl_n>`
     - :python:`float`
     - mm
     - effective focal length, scaled by the refractive index
   
   * - :attr:`bfl <optrace.tracer.transfer_matrix_analysis.TMA.bfl>`
     - :python:`float`
     - mm
     - back focal length

   * - :attr:`ffl <optrace.tracer.transfer_matrix_analysis.TMA.ffl>`
     - :python:`float`
     - mm
     - front focal length

   * - :attr:`wl <optrace.tracer.transfer_matrix_analysis.TMA.wl>`
     - :python:`float`
     - nm
     - wavelength for the analysis

Access the properties in the following way:

.. doctest::

   >>> tma.efl
   30.645525910383494

.. doctest::

   >>> tma.abcd
   array([[ 0.9046767 ,  6.50763158],
          [-0.03263119,  0.87064057]])


Calculating Image and Object Distance
__________________________________________

The method :meth:`image_position <optrace.tracer.transfer_matrix_analysis.TMA.image_position>` allows for the calculation of an image position.
You need to provide an object position:

.. doctest::

   >>> tma.image_position(-50)
   72.87925720752206

Both input and output value are absolute positions at the optical axis in millimeters.

Conversely, we can calculate an object position from a known image position with :meth:`object_position <optrace.tracer.transfer_matrix_analysis.TMA.object_position>`:

.. doctest::

   >>> tma.object_position(100)
   -33.84654855214077

In botch cases infinite values (:python:`-np.inf, np.inf`) are supported as function parameters.
For the image position at infinity we get:

.. doctest::

   >>> tma.object_position(np.inf)
   -16.93123809931588

Which is equal to the position of the first focal point:

.. doctest::
   
   >>> tma.focal_points[0]
   -16.93123809931588


Analogously the magnification factors at the image/object plane can be calculated:

.. doctest::

   >>> tma.image_magnification(-57.3)
   -0.7591396036811361

.. doctest::

   >>> tma.object_magnification(18)
   0.8640542105175426

A positive factor corresponds to an upright image, a negative to an inverted one. 
A magnitude larger than one implies magnification, a smaller number a size decrease.

Details on the implementation are described in :numref:`ray_image_object_distances`.

Another feature is the calculation of the ABCD matrix for a specific object and image distance.
The corresponding :meth:`matrix_at <optrace.tracer.transfer_matrix_analysis.TMA.matrix_at>` method requires the object and image position:

.. doctest::

   >>> tma.matrix_at(-60, 80.2)
   array([[ -1.16560585, -19.55567495],
          [ -0.03263119,  -1.40538498]])


Calculation of Entrance and Exit Pupils
__________________________________________

Methods for calculating the entrance and exit pupil position and magnifications are also available.
Details on the math are found in :numref:`pupil_calculation`.

The following example loads the paraxial eye model from :func:`legrand_eye() <optrace.tracer.presets.geometry.legrand_eye>` and creates the TMA object:

.. testcode::

   eye = ot.presets.geometry.legrand_eye()
   aps = eye.apertures[0].pos[2]
   tma = eye.tma()

The function :meth:`pupil_position <optrace.tracer.transfer_matrix_analysis.TMA.pupil_position>` requires an aperture stop position argumentand returns a tuple of entrance and exit pupil position along the optical axis.
The aperture can lie inside, behind or in front of the lens setup. 

.. doctest::
   
   >>> tma.pupil_position(aps)
   (3.037565216550855, 3.6821114369501466)

The method :meth:`pupil_magnification <optrace.tracer.transfer_matrix_analysis.TMA.pupil_magnification>` calculates the pupil magnifications:

.. doctest::
   
   >>> tma.pupil_magnification(aps)
   (1.1310996628960361, 1.0410557184750733)


Miscellaneous Properties
__________________________________________

The calculation is currently limited to these properties.
Unfortunately pupil sizes, numerical apertures, f-numbers, airy disk diameters are not available.
This is due to the |TMA_link| object not having any information about the lens diameters or ray characteristics.

In some cases the properties can be estimated using the interactive GUI and ray picking.
For instance, the pupil sizes can be calculated from the pupil positions from the |TMA_link| and the radial distance of the outermost traced rays at this position.

