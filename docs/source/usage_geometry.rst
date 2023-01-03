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
 * **RaySource**: An element with a light emitting surface
 * **Lens**: An element with two surfaces on which light is refracted.
 * **IdealLens**: Like a Lens, except that it has a planar surface and refracts light without aberration
 * **Detector**: Element with one surface on which images can be rendered
 * **Filter**: Element with a surface on which wavelength-dependent or wavelength-independent filtering takes place.
 * **Aperture**: Like a filter, except that incident light is completely absorbed.
 * **Marker**: element consisting of a point and a label, useful for annotating things.

These subclasses have the same methods as the Element superclass, these include:
 * move the element: 
    * :python:`El.move_to([-2.1, 0.2, 5.3])`
 * rotate: 
    * :python:`El.rotate(25)`
 * flip around the x-axis: 
    * :python:`El.flip()`
 * getting the extent (bounding box): 
    * :python:`ext = El.extent`
 * determine the position: 
    * :python:`pos = El.pos`
 * plot the geometry


Group
________________

A *Group* can be seen as a list or container of several elements.

It contains the following functionality:
 * Adding and removing one or more elements:
    * :python:`G.add(obj)`
    * :python:`G.remove(obj)`
 * Emptying all elements: 
    * :python:`G.clear()`
 * check if an element is included: 
    * :python:`G.has(obj)`
 * move all elements at once: 
    * :python:`G.move_to([5.3, 0.0, 12.3])`
 * rotate or flip all elements: 
    * :python:`G.rotate(-12)`
    * :python:`G.flip()`
 * create ray transfer matrix of the whole lens system: 
    * :python:`G.tma()`


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

.. TODO



Geometry Presets
_______________________




Loading ZEMAX Geometries (.zmx)
__________________________________


