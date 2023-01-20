Base Geometries (Element, Group, Raytracer)
------------------------------------------------

.. testsetup:: *

   import optrace as ot

.. role:: python(code)
  :language: python
  :class: highlight

Elements
__________________

In ``optrace`` the class *Element* denotes an object which has no, one or two surfaces and belongs to the tracing geometry.

This includes the classes

.. list-table::
   :widths: 100 400
   :header-rows: 0
   :align: left

   * - **RaySource**
     - An element with a light emitting surface
   * - **Lens**
     - An element with two surfaces on which light is refracted.
   * - **IdealLens**
     - Like a Lens, except that it has a planar surface and refracts light without aberration
   * - **Detector**
     - Element with one surface on which images can be rendered
   * - **Filter** 
     - Element with a surface on which wavelength-dependent or wavelength-independent filtering takes place.
   * - **Aperture** 
     - Like a filter, except that incident light is completely absorbed.
   * - **Marker** 
     - element consisting of a point and a label, useful for annotating things.

These subclasses have the same methods as the Element superclass, these include:

.. list-table::
   :header-rows: 1
   :align: left

   * - Functionality
     - Example
   * - move the element: 
     - :python:`El.move_to([-2.1, 0.2, 5.3])`
   * - rotate 
     - :python:`El.rotate(25)`
   * - flip around the x-axis: 
     - :python:`El.flip()`
   * - getting the extent (bounding box): 
     - :python:`ext = El.extent`
   * - determine the position: 
     - :python:`pos = El.pos`
   * - plot the geometry
     - (internal functions)


Group
________________

A *Group* can be seen as a list or container of several elements.

It contains the following functionality:

.. list-table::
   :widths: 300 250
   :header-rows: 1
   :align: left

   * - Functionality
     - Example
   * - Adding and removing one or more elements:
     - | :python:`G.add(obj)`
       | :python:`G.remove(obj)`
   * - Emptying all elements: 
     - :python:`G.clear()`
   * - check if an element is included: 
     - :python:`G.has(obj)`
   * - move all elements at once: 
     - :python:`G.move_to([5.3, 0.0, 12.3])`
   * - rotate or flip all elements: 
     - | :python:`G.rotate(-12)`
       | :python:`G.flip()`
   * - create ray transfer matrix of the whole lens system: 
     - :python:`G.tma()`

A Group object stores all elements in their own class lists:
``lenses, ray_sources, detectors, markers, filters, apertures``.
Where IdealLens and Lens are included in the same list.

When adding objects, the order of objects remains the same.
Thus ``lenses[2]`` denotes the lens that was added third (since counting starts at 0).
In principle it is recommended to add objects in the order in which the light passes through them.

.. TODO example

Raytracer
________________


The raytracer class provides the functionality for tracing, geometry checking, rendering spectra and images, and focusing.

Since the raytracer is a subclass of a group, elements can be changed or added in the same way.


.. TODO Screenshot einer Raytracer Geometry in der GUI


**Outline**

All objects and rays can only exist in a three-dimensional box, the *outline*.
When initializing the raytracer this is passed as ``outline`` parameter.
This is also the only mandatory parameter of this class


.. testcode::

   RT = ot.Raytracer(outline=[-2, 2, -3, 3, -5, 60])



**Geometry**

Since ``optrace`` implements sequential raytracing, the surfaces and objects must be in a well-defined and unique sequence. This applies to all elements with interactions of light: ``Lens, IdealLens, Filter, Aperture, RaySource``.
The elements ``Detector, Marker`` are excluded from this.
All RaySource elements must lie before all lenses, filters and apertures. And all subsequent lenses, filters, apertures must not collide and be inside the outline.


**Surrounding Media**

Earlier we learned that when creating a lens, you can use the ``n2`` parameter to define the subsequent media. In the case of multiple lenses, the ``n2`` of the previous lens is the medium before the next lens.
In the case of the raytracer, we can define an ``n0`` which defines the refractive index for all undefined ``n2=None`` as well as for the region to the first lens.

.. TODO Prinzipbild mit mehreren Linsen und Medienübergängen

**absorb_missing**

The ``absorb_missing`` parameter, which is set to ``True`` by default, ensures that light which does not hit a lens is absorbed. In principle, this is the typical and desired case. However, there are geometries where ``absorb_missing=False`` could be useful. 

A special case is when a ray does not hit a lens where a transition from surrounding media takes place. Here the rays are absorbed in any case, because the exact transition geometry is defined only at the lens itself.


**no_pol**

The raytracer provides the functionality to trace polarization directions. Thus, not only the polarization vector for the ray and ray segment can be calculated, but also the exact transmission at each surface transition.
Unfortunately, the calculation is comparatively computationally intensive.

With the parameter ``no_pol=True`` no polarizations are calculated and we assume unpolarized/uniformly polarized light at each transmission. Typically this speeds up the tracing by 10-30%.
Whether you can neglect the influence of polarization depends of course on the exact setup of the geometry.
However, for setups where the angles of the beams to surface normals are small, this is usually the case.


**Example**

Below you can find an example. A eye preset is loaded and flipped around the x-axis.
A point source is added at the retina and the geometry is traced.

.. testcode::

   import optrace as ot

   # init raytracer 
   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

   # load eye preset
   eye = ot.presets.geometry.arizona_eye(pupil=3)

   # flip, move and add it to the tracer
   eye.flip()
   eye.move_to([0, 0, 0])
   RT.add(eye)

   # create and add divergent point source
   point = ot.Point()
   RS = ot.RaySource(point, spectrum=ot.presets.light_spectrum.d50, divergence="Isotropic", div_angle=5,
                     pos=[0, 0, 0])
   RT.add(RS)

   # trace
   RT.trace(100000)

.. testoutput::
   :hide:

   Class ...

Loading ZEMAX Geometries (.zmx)
__________________________________


It is possible to load ``.zmx`` geometries into ``optrace``. For instance, the following example load some geometry from file ``setup.zmx`` into the raytracer.

.. code-block:: python

   RT = ot.Raytracer(outline=[-20, 20, -20, 20, -20, 200])

   RS = ot.RaySource(ot.CircularSurface(r=0.05), spectrum=ot.presets.light_spectrum.d65, pos=[0, 0, -10])
   RT.add(RS)

   n_schott = ot.load.agf("schott.agf")
   G = ot.load.zmx("setup.zmx", n_dict=n_schott)
   RT.add(G)

   RT.trace(10000)


For the materials to be loaded correctly all mentioned names in the ``.zmx`` file need to be included in the ``n_dict`` dictionary.
You can either load them from a ``.agf`` catalogue like in :numref:`agf_load` or create the dictionary manually.

A list of exemplary ``.zmx`` files can be found in the following `repository <https://github.com/nzhagen/LensLibrary/tree/main/zemax_files>`_.


Unfortunately, the support is only experimental, as there is no actual documentation on the file format. Additionally, only a subset of all ZEMAX functionality is supported, including:

* ``SEQ``-mode only
* ``UNIT`` must be ``MM``
* only ``STANDARD`` or ``EVENASPH`` surfaces, this is equivalent to ``RingSurface, CircularSurface, SphericalSurface, ConicSurface, AspheriSurface`` in ``optrace``
* no support for coatings
* temperature or absorption behavior of the material is neglected
* only loads lens and aperture geometries, no support for additional objects

Information on the file format can be found `here <https://documents.pub/document/zemaxmanual.html?page=461>`__, `here <https://github.com/mjhoptics/ray-optics/blob/master/src/rayoptics/zemax/zmxread.py>`__ and `here <https://github.com/quartiq/rayopt/blob/master/rayopt/zemax.py>`__.


Geometry Presets
_______________________


**LeGrand Paraxial Eye Model**

The LeGrand full theoretical eye model is a simple model consisting of only spherical surfaces and wavelength-independent refractive indices. It models the paraxial behavior of a far-adapted eye.

.. list-table:: LeGrand Full Theoretical Eye Model :footcite:`SchwiegerlingOptics`
   :widths: 110 75 75 75 75
   :header-rows: 1
   :align: center

   * - Surface
     - Radius in mm
     - Conic Constant
     - Refraction Index to next surface
     - Thickness (mm) (to next surface)

   * - Cornea Anterior
     - 7.80
     - 0
     - 1.3771
     - 0.5500
		
   * - Cornea Posterior 
     - 6.50
     - 0 
     - 1.3374
     - 3.0500

   * - Lens Anterior 
     - 10.20
     - 0
     - 1.4200
     - 4.0000

   * - Lens Posterior 
     - -6.00
     - 0 
     - 1.3360
     - 16.5966

   * - Retina 
     - -13.40
     - 0 
     - `-` 
     - `-`


The preset is located in ``ot.presets.geometry`` and is called as function. It returns a ``Group`` object that can be added to a raytracer. Provide a ``pos`` parameter to position it at an other position than ``[0, 0, 0]``.

.. testcode::

   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])
   eye_model = ot.presets.geometry.legrand_eye(pos=[0.3, 0.7, 1.2])
   RT.add(eye_model)

Optional parameters include a pupil diameter and a detector (retina) radius, both provided in millimeters.

.. testcode::

   eye_model = ot.presets.geometry.legrand_eye(pupil=3, r_det=10, pos=[0.3, 0.7, 1.2])


**Arizona Eye Model**

A more advanced model is the Arizona Eye Model, which tries to match clinical levels of aberration and for different adaption levels. It consists of conic surfaces, dispersive media and adaptation dependent parameters.

.. list-table:: Arizona Eye Model :footcite:`SchwiegerlingOptics`
   :widths: 75 75 75 75 75 75
   :header-rows: 1
   :align: center

   * - Surface
     - Radius in mm
     - Conic Constant
     - Refraction Index to next surface
     - Abbe Number
     - Thickness (mm) (to next surface)

   * - Cornea Anterior
     - 7.80
     - -0.25
     - 1.377
     - 57.1
     - 0.55
		
   * - Cornea Posterior 
     - 6.50
     - -0.25
     - 1.337
     - 61.3
     - :math:`t_\text{aq}`

   * - Lens Anterior 
     - :math:`R_\text{ant}`
     - :math:`K_\text{ant}`
     - :math:`n_\text{lens}`
     - 51.9
     - :math:`t_\text{lens}`

   * - Lens Posterior 
     - :math:`R_\text{post}`
     - :math:`K_\text{post}`
     - 1.336
     - 61.1
     - 16.713

   * - Retina 
     - -13.40
     - 0 
     - `-` 
     - `-` 

     - `-` 

With an accommodation level :math:`A` in dpt the missing parameters are calculated using: :footcite:`SchwiegerlingOptics`

.. math::
   \begin{array}{ll}
       R_{\text {ant }}=12.0-0.4 A & K_{\text {ant }}=-7.518749+1.285720 A \\
       R_{\text {post }}=-5.224557+0.2 A & K_{\text {post }}=-1.353971-0.431762 A \\
       t_{\text {aq }}=2.97-0.04 A & t_{\text {lens }}=3.767+0.04 A \\
       n_{\text {lens }}=1.42+0.00256 A-0.00022 A^2
   \end{array}


Accessing and adding works like for the ``legrand_eye`` preset.

.. testcode::

   RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])
   eye_model = ot.presets.geometry.arizona_eye(pos=[0.3, 0.7, 1.2])
   RT.add(eye_model)

As for the legrand eye we have the parameters ``pupil`` and ``r_det``. Additionally there is a ``accommodation`` parameter specified in diopters, which defaults to 0 dpt.

.. testcode::

   eye_model = ot.presets.geometry.arizona_eye(adaptation=1, pupil=3, r_det=10, pos=[0.3, 0.7, 1.2])


.. figure:: ./images/arizona_eye_scene.png
   :align: center
   :width: 600

   Eye model in the ``arizona_eye_model.py`` example script.

------------

**Sources**

.. footbibliography::


