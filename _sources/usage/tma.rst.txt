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

A ray transfer matrix analysis object |TMA_link| allows for the analysis of the paraxial properties of a lens or lens setups. After being computed once and for a specific wavelength it stores a snapshot of different quantities.
These for instance include focal lengths, nodal points, optical powers and more.

You can read about the calculation and meaning of the matrix analysis in :numref:`ray_matrix_analysis`.

Defining the Geometry for the Analysis
__________________________________________

**Group/Raytracer**

From a :class:`Raytracer <optrace.tracer.raytracer.Raytracer>` with a specific geometry the |TMA_link| objected can be created with the member function :meth:`tma() <optrace.tracer.geometry.group.Group.tma>`.

.. testcode::

   tma = RT.tma()

The function optionally takes a wavelength in nanometers as argument, for which the properties are calculated.

.. testcode::

   tma = RT.tma(780)

For a Group object this everything works the same way:

.. testcode::

   G = ot.presets.geometry.arizona_eye()
   tma = G.tma()


**Single Lens**

We can also create the analysis object for a single lens:

.. testcode::

   front = ot.CircularSurface(r=5)
   back = ot.SphericalSurface(r=5, R=-25)
   n = ot.presets.refraction_index.K5
   L = ot.Lens(front, back, n=n, d=0.5, pos=[0, 0, 10])

   tma = L.tma()

But since a single Lens does not know about the ambient medium before it, we can provide it as parameter :python:`n0` to calculate the properties correctly for a non-default (different than vacuum) prior medium.

.. testcode::

   n0 = ot.RefractionIndex("Constant", n=1.1)
   tma = L.tma(n0=n0)

**Multiple Lenses**

Without a specific geometry we can also create the |TMA_link| object by simply providing a list of lenses.

.. testcode::

   back2 = ot.SphericalSurface(r=5, R=-25)
   front2 = ot.CircularSurface(r=5)
   n2 = ot.presets.refraction_index.F2
   L2 = ot.Lens(front, back, n=n2, de=0.5, pos=[0, 0, 16])

   Ls = [L, L2]
   tma = ot.TMA(Ls)

As for the lens the ambient medium before the first lens is not known but can be provided with the :python:`n0` parameter.

.. testcode::

   tma = ot.TMA(Ls, n0=n0)

Paraxial Properties
__________________________________________


Below a tabular overview of the supported properties is found. Details on their meaning and calculation are documented in :numref:`ray_cardinal_points` and more information on the different definitions for focal lengths and powers in :numref:`ray_power_def`.

.. list-table:: Properties of a |TMA_link| object
   :widths: 75 60 40 200
   :header-rows: 1
   :align: center

   * - Variable
     - Type
     - Unit
     - Meaning

   * - :attr:`n1 <optrace.tracer.transfer_matrix_analysis.TMA.n1>`
     - float
     - ``-``
     - refractive index value before the lens setup
   
   * - :attr:`n2 <optrace.tracer.transfer_matrix_analysis.TMA.n2>`
     - float
     - ``-``
     - refractive index value after the lens setup
   
   * - :attr:`vertex_points <optrace.tracer.transfer_matrix_analysis.TMA.vertex_points>`
     - float, float (tuple)
     - mm
     - front and back position of vertices of the system 
   
   * - :attr:`d <optrace.tracer.transfer_matrix_analysis.TMA.d>`
     - float
     - mm
     - thickness, distance between vertex points
   
   * - :attr:`abcd <optrace.tracer.transfer_matrix_analysis.TMA.abcd>`
     - numpy array, shape (2, 2)
     - ``-``
     - ABCD matrix

   * - :attr:`principal_points <optrace.tracer.transfer_matrix_analysis.TMA.principal_points>`
     - float, float (tuple)
     - mm
     - principal points (z-positions)

   * - :attr:`nodal_points <optrace.tracer.transfer_matrix_analysis.TMA.nodal_points>`
     - float, float (tuple)
     - mm
     - nodal points (z-positions)
   
   * - :attr:`optical_center <optrace.tracer.transfer_matrix_analysis.TMA.optical_center>`
     - float
     - mm
     - optical center (z-position)
   
   * - :attr:`focal_points <optrace.tracer.transfer_matrix_analysis.TMA.focal_points>`
     - float, float (tuple)
     - mm
     - focal points (z-positions)
   
   * - :attr:`focal_lengths <optrace.tracer.transfer_matrix_analysis.TMA.focal_lengths>`
     - float, float (tuple)
     - mm
     - focal lengths
   
   * - :attr:`focal_lengths_n <optrace.tracer.transfer_matrix_analysis.TMA.focal_lengths_n>`
     - float, float (tuple)
     - mm
     - focal lengths, scaled with refractive index

   * - :attr:`powers <optrace.tracer.transfer_matrix_analysis.TMA.powers>`
     - float, float (tuple)
     - dpt
     - optical powers of the system
   
   * - :attr:`powers_n <optrace.tracer.transfer_matrix_analysis.TMA.powers_n>`
     - float, float (tuple)
     - dpt
     - optical powers, scaled with the refractive index
   
   * - :attr:`efl <optrace.tracer.transfer_matrix_analysis.TMA.efl>`
     - float
     - mm
     - effective focal length of the system

   * - :attr:`efl_n <optrace.tracer.transfer_matrix_analysis.TMA.efl_n>`
     - float
     - mm
     - effective focal length, scaled by the refractive index
   
   * - :attr:`bfl <optrace.tracer.transfer_matrix_analysis.TMA.bfl>`
     - float
     - mm
     - back focal length

   * - :attr:`ffl <optrace.tracer.transfer_matrix_analysis.TMA.ffl>`
     - float
     - mm
     - front focal length

   * - :attr:`wl <optrace.tracer.transfer_matrix_analysis.TMA.wl>`
     - float
     - nm
     - wavelength for the analysis


The above properties can be simply accessed like the following examples:

.. doctest::

   >>> tma.efl
   30.645525910383494

.. doctest::

   >>> tma.abcd
   array([[ 0.9046767 ,  6.50763158],
          [-0.03263119,  0.87064057]])


Calculating Image and Object Distance
__________________________________________


The member function :meth:`image_position <optrace.tracer.transfer_matrix_analysis.TMA.image_position>` enables us to calculate a image position from an object position.

.. doctest::

   >>> tma.image_position(-50)
   72.87925720752206

Both input and output value are absolute positions at the optical axis in millimeters.

On the contrary we can calculate an object position from a known image position with
:meth:`object_position <optrace.tracer.transfer_matrix_analysis.TMA.object_position>`:

.. doctest::

   >>> tma.object_position(100)
   -33.84654855214075

For both function infinite values (:python:`-np.inf, np.inf`) are supported as function parameters.
For the image position at infinity we get:

.. doctest::

   >>> tma.object_position(np.inf)
   -16.931238099315877

Which should be exactly the same position as the first focal point:

.. doctest::
   
   >>> tma.focal_points[0]
   -16.93123809931588



Analogously not only the positions but also the magnification factors at the image/object plane can be calculated:

.. doctest::

   >>> tma.image_magnification(-57.3)
   -0.7591396036811361

.. doctest::

   >>> tma.object_magnification(18)
   0.8640542105175426

A positive factor corresponds to an upright image, a negative to an inverted one. A number of magnitude larger than one means magnification, a number smaller than this a size decrease.

Details on the math are listed in :numref:`ray_image_object_distances`.

Another feature is the calculation of the ABCD matrix for a specific object and image distance.
The :meth:`matrix_at <optrace.tracer.transfer_matrix_analysis.TMA.matrix_at>` method takes the object and image position as arguments and returns the matrix.

.. doctest::

   >>> tma.matrix_at(-60, 80.2)
   array([[ -1.16560585, -19.55567495],
          [ -0.03263119,  -1.40538498]])


Calculation of Entrance and Exit Pupil
__________________________________________

Entrance and exit pupil position and magnifications are also available for calculation.
Details on the math are found in :numref:`pupil_calculation`.

First, let's load the paraxial eye model, get the pupil position and create the matrix analysis object:

.. testcode::

   eye = ot.presets.geometry.legrand_eye()
   aps = eye.apertures[0].pos[2]
   tma = eye.tma()

The function :meth:`pupil_position <optrace.tracer.transfer_matrix_analysis.TMA.pupil_position>` takes the aperture stop position and returns a tuple of entrance and exit pupil position along the optical axis.
Regarding the position of the stop, the aperture can lie inside, behind or in front of the lens setup. Therefore there are no limitations.

.. doctest::
   
   >>> tma.pupil_position(aps)
   (3.0375652165508553, 3.6821114369501466)

Magnifications are returned with the member function :meth:`pupil_magnification <optrace.tracer.transfer_matrix_analysis.TMA.pupil_magnification>`.

.. doctest::
   
   >>> tma.pupil_magnification(aps)
   (1.1310996628960361, 1.0410557184750733)

