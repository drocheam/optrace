Future Features
-----------------


TRA, LRA calculations and plots
_____________________________________

see https://www.eckop.com/resources/optical-testing/measuring-aberrations/

modification of ``hit_detector()``, such that:

* only works on one source
* only works on sources with one degree of freedom (Line: Line Parameter, Point: Opening Angle)
* returns TRA/LRA values

Problems:
 * only allow discrete wavelengths?
 * plot separate curves?

Geometry Optimization
_________________________

* Building helper methods, that allow for a simple optimization
* cost function handling, as well as input parameter ranges
* wrapper for different optimization methods from ``scipy.optimize``

Problems: how to handle invalid geometries or errors?

