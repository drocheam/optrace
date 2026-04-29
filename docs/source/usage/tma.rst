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

A ray transfer matrix analysis object, the |TMA_link|, provides paraxial analysis features for a lens or lens setup.
Available properties include focal lengths, nodal points, optical powers and more.
They are calculated for a specific wavelength and the current state of the geometry.

Defining the Analysis Geometry
__________________________________________

**Group/Raytracer**

From a :class:`Raytracer <optrace.tracer.raytracer.Raytracer>`, the |TMA_link| for the full geometry
is calculated with the method :meth:`tma() <optrace.tracer.geometry.group.Group.tma>`.

.. testcode::

   tma = RT.tma()

The function optionally takes a wavelength in nanometers as argument, for which the properties are calculated:

.. testcode::

   tma = RT.tma(780)

Analyzing a :class:`Group <optrace.tracer.geometry.group.Group>` follows the same procedure:

.. testcode::

   G = ot.presets.geometry.arizona_eye()
   tma = G.tma()


**Single Lens**

The analysis is also available for a single :class:`Lens <optrace.tracer.geometry.lens.Lens>`:

.. testcode::

   front = ot.CircularSurface(r=5)
   back = ot.SphericalSurface(r=5, R=-25)
   n = ot.presets.refraction_index.K5
   L = ot.Lens(front, back, n=n, d=0.5, pos=[0, 0, 10])

   tma = L.tma()

While a single Lens defines the subsequent ambient medium through :python:`Lens.n2`, 
it has no knowledge on the preceding one.
Normally it would be determined by either the :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` 
or the previous lens.
The same applies to the :python:`n0` medium of the Raytracer, 
which defines all otherwise unspecified :python:`Lens.n2` media
To explicitly define it for the TMA, we can provide a :python:`n0` parameter to it.
Otherwise it defaults to :math:`n = 1`.

.. testcode::

   n0 = ot.RefractionIndex("Constant", n=1.1)
   tma = L.tma(n0=n0)

**Multiple Lenses**

Without a specific geometry, the |TMA_link| can also be created by providing a list of lenses.

.. testcode::

   back2 = ot.SphericalSurface(r=5, R=-25)
   front2 = ot.CircularSurface(r=5)
   n2 = ot.presets.refraction_index.F2
   L2 = ot.Lens(front, back, n=n2, de=0.5, pos=[0, 0, 16])

   Ls = [L, L2]
   tma = ot.TMA(Ls)

As with a single lens, set the previous ambient medium (and the medium for all undefined :python:`Lens.n2`) 
the :python:`n0` parameter.

.. testcode::

   tma = ot.TMA(Ls, n0=n0)


Paraxial Properties
__________________________________________

The following table provides an overview of supported TMA properties.
Details on their definition and calculation are documented in :numref:`ray_matrix_analysis`, 
while more information on the different definitions for focal lengths and powers are found in :numref:`ray_power_def`.

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

Accessing the properties as follows:

.. doctest::

   >>> tma.efl
   30.645525910383494

.. doctest::

   >>> tma.abcd
   array([[ 0.9046767 ,  6.50763158],
          [-0.03263119,  0.87064057]])


Image and object distance, as well as entrance and exit pupil, 
are available through the methods described further down below.

Calculating Image and Object Distance
__________________________________________

The method :meth:`image_position <optrace.tracer.transfer_matrix_analysis.TMA.image_position>` 
calculates the image position for a given object position:

.. doctest::

   >>> tma.image_position(-50)
   72.87925720752206

Both input and output values are absolute positions in millimeters at the optical axis.

Conversely, an object position from a known image position is calculated 
with :meth:`object_position <optrace.tracer.transfer_matrix_analysis.TMA.object_position>`:

.. doctest::

   >>> tma.object_position(100)
   -33.84654855214077

In both cases, infinite values (:python:`-np.inf, np.inf`) are valid inputs to the function:

.. doctest::

   >>> tma.object_position(np.inf)
   -16.93123809931588

It is equivalent to the position of the first focal point:

.. doctest::
   
   >>> tma.focal_points[0]
   -16.93123809931588

Analogously, the magnification factors at the image/object plane are found with:

.. doctest::

   >>> tma.image_magnification(-57.3)
   -0.7591396036811361

.. doctest::

   >>> tma.object_magnification(18)
   0.8640542105175426

A positive factor corresponds to an upright image, a negative to an inverted one. 
A magnitude larger than one implies magnification, a smaller number a size decrease.

Details on the implementation are described in :numref:`ray_image_object_distances`.

Besides `TMA.abcd <optrace.tracer.transfer_matrix_analysis.TMA.abcd>`,
which defines the system from vertex to vertex,
the class also provides a convenient :meth:`matrix_at <optrace.tracer.transfer_matrix_analysis.TMA.matrix_at>`,
which calculates the ABCD matrix for an additional given object and image distance:

.. doctest::

   >>> tma.matrix_at(-60, 80.2)
   array([[ -1.16560585, -19.55567495],
          [ -0.03263119,  -1.40538498]])


Calculation of Entrance and Exit Pupils
__________________________________________

Methods for calculating the entrance and exit pupil position and magnifications are also available.
Details on their computation are found in :numref:`pupil_calculation`.

The following example loads the paraxial eye model from 
:func:`legrand_eye() <optrace.tracer.presets.geometry.legrand_eye>` and creates the TMA object:

.. testcode::

   eye = ot.presets.geometry.legrand_eye()
   aps = eye.apertures[0].pos[2]
   tma = eye.tma()

The method :meth:`pupil_position <optrace.tracer.transfer_matrix_analysis.TMA.pupil_position>` 
requires an aperture stop position argument 
and returns a tuple of entrance and exit pupil position along the optical axis.
The provided aperture position can lie inside, behind or in front of the optical system. 

.. doctest::
   
   >>> tma.pupil_position(aps)
   (3.037565216550855, 3.6821114369501466)

The method :meth:`pupil_magnification <optrace.tracer.transfer_matrix_analysis.TMA.pupil_magnification>` 
calculates the corresponding pupil magnifications:

.. doctest::
   
   >>> tma.pupil_magnification(aps)
   (1.1310996628960361, 1.0410557184750733)


Limitations
__________________________________________

Pupil sizes, numerical apertures, f-numbers and airy disk diameters are not available 
due to current limitations of the |TMA_link| design.

