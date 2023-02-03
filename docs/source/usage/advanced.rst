Advanced Topics
------------------------------------------------


Raytracing Errors
_________________________


Accessing Ray Properties
_____________________________


Overview
################

**Terms:**

| **Ray Section**: Part of the ray, where the direction is constant. Sections typically start and end at surface intersections.
| **Ray**: sum of all its ray sections, entirety of the ray going from the source to the point it is absorbed


**Shapes**:

| **N**: number of rays
| **nt**: number of sections per ray, equal for all rays


The number of sections is the same for all rays. If a ray gets absorbed early, all consecutive sections consist of zero length vectors starting at the last position and having their power set to zero. Direction and polarization are undefined.


.. list-table:: List of ray properties
   :widths: 100 200 50 400
   :header-rows: 0
   :align: left

   * - Name
     - Type
     - Unit
     - Function
   * - ``p_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x nt x 3
     - mm
     - 3D starting position for all ray sections 
   * - ``s0_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x 3
     - ``-``
     - unity direction vector at the ray source
   * - ``pol_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N x nt x 3
     - ``-``
     - unity 3D polarization vector
   * - ``w_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N x nt
     - W
     - ray power
   * - ``n_list``
     - ``np.ndarray`` of type ``np.float64`` of shape N x nt
     - ``-``
     - refractive indices for all ray sections
   * - ``wl_list``
     - ``np.ndarray`` of type ``np.float32`` of shape N
     - nm
     - wavelength of the ray
    

Direct Access
################


Masking
################


Controlling Threading and Standard Output
______________________________________________


Modifying Initialized Objects
____________________________________________



