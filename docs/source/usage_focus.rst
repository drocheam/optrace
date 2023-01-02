Autofocus
-----------------------

Goal
____________________

Focus finding can be categorized into two different goals

1. finding a focal point
2. finding the position of an image plane in an imaging system


Focus Modes
____________________

The following focus methods are available:

* **Position Variance**: minimizes the variance of the lateral ray position
* **Airy Disc Weighting**: weights the ray positions with a spatial sensitivity of the zeroth order of an airy disc
* **Irradiance Maximum**: finds the position with the highest irradiance
* **Irradiance Variance**: finds the positions with the highest irradiance variance
* **Image Sharpness**: finds the position with the sharpest edges

More details as well as the mathematical formulations can be found in :numref:`autofocus`.

Application Cases
____________________

Below you can find multiple application cases an preferred autofocus methods.

**Case 1**: perfect, ideal focal point
 * **examples:** focus of an ideal lens. Small, local illumination of a real lens
 * **preferred methods:** all methods find the focus correctly, for performance reason "Position Variance" should be used

**Case 2:**  broad or no distinct focal point
 * **examples:** lens with large spherical aberration, multifocal lens
 * **preferred methods:** None, largely different behavior depending on method choice
 * **behaviour known from experience**
    * Position Variance: finds a compromise between multiple foci, often inbetween their position
    * Airy Disc Weighting: Ignores glares, halos and rays with large distance from airy disc
    * Irradiance Maximum: finds the focus with the largest irradiance
    * Image Sharpness: Not suited, since its searches for sharp structures
    * Irradiance Variance: similar behavior to Image Sharpness and Irradiance Maximum

**Case 3:** finding the image distance
 * **example:** lens setup with multiple lenses, we want to find the distance where the image has the highest sharpness
 * **preferred methods:** Image Sharpness, in some specific edge cases Irradiance Variance/Maximum might work.


.. topic:: Note

   Generally it is recommended to plot the cost function of the optimization so one can see if there are multiple minima and how distinct the found value is.
   The TraceGUI has an option for plotting the cost function.


Debug Plots
___________________________


