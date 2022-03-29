
"""
Surface class:
Provides the functionality for the creation of numerical or analytical surfaces.
The class contains methods for interpolation, surface masking and normal calculation
"""


import copy
import numpy as np
import Backend.Misc as misc
from Backend.SurfaceFunction import *

# TODO Mode Data: check if working

class Surface:

    def __init__(self,
                 surface_type:  str,
                 r:             float = 3.,
                 rho:           float = 1/10,
                 k:             float = -0.444,
                 ri:            float = 0.1,
                 dim:           (list | np.ndarray) = [6, 6],
                 ang:           float = 0,
                 Data:          np.ndarray = None,
                 func:          SurfaceFunction = None)\
            -> None:
        """
        Create a surface object.
        The z-coordinate in the pos array is irrelevant if the surface is used for a lens, since it wil be
        adapted inside the lens class

        :param surface_type: "Asphere", "Sphere", "Circle", "Ring" or "Data" (string)
        :param rho: curvature constant (=1/R) for surface_type="Asphere" or "Sphere" (float)
        :param r: radial size for surface_type="Asphere", "Sphere", "Circle" or "Ring" (float)
        :param k: conic constant for surface_type="Asphere" (float)
        :param dim:
        :param ang:
        :param Data: uses copy of this array, grid z-coordinate array for surface_type="Data" (numpy 2D array)
        :param ri: radius of inner circle for surface_type="Ring" (float)
        :param func:
        """

        self.surface_type = surface_type

        self.pos = np.array([0., 0., 0.], dtype=np.float64)
        self.dim = np.array(dim, dtype=np.float64)

        self.r, self.ri = float(r), float(ri)
        self.rho, self.k = float(rho), float(k)
        self.ang = float(ang)
        self.func = func

        self.eps = max(self.r / 1e6, 1e-12)

        if r < 0:
            raise ValueError("Surface radius r needs to be positive.")

        if ri < 0:
            raise ValueError("Inner Surface radius ri needs to be non-negative.")

        if not isinstance(func, SurfaceFunction | None):
            raise TypeError("func needs to be a SurfaceFunction object.")

        if dim[0] <= 0 or dim[1] <= 0:
            raise ValueError("Dimensions dim need to be positive.")

        if self.rho == 0:
            raise ValueError("rho needs to be non-zero. For a plane use surface_type=\"Circle\".")


        match surface_type:

            case ("Asphere" | "Sphere"):

                if surface_type == "Sphere":
                    self.k = 0.

                # (k+1) * rho**2 * r**2 = 1 defines the edge of the conic section
                # we can only calculate it inside this area
                # sections with negative k have no edge
                if (self.k + 1)*(self.rho*self.r)**2 >= 1:
                    raise ValueError("Surface radius r larger than radius of conic section.")

                self.maxz = None

                self.minz = self.pos[2] if self.rho > 0 \
                                    else self.getValues(np.array([r+self.pos[0]]), np.array([self.pos[1]]))[0]
                self.maxz = self.pos[2] if self.rho < 0 \
                                    else self.getValues(np.array([r+self.pos[0]]), np.array([self.pos[1]]))[0]

            case "Data":
   
                self.Z = np.array(Data, dtype=np.float64)
                self.Mask = np.isfinite(self.Z)
                self.Z = misc.interpolateNan(self.Z)

                ny, nx = self.Z.shape
                
                if nx != ny:
                    raise ValueError("Matrix 'Data' needs to have a square shape.")
    
                self.xy = np.linspace(-self.r, self.r, nx)
                self.eps = (self.xy[1] - self.xy[0]) / 1e3

                self.minz, self.maxz = np.nanmin(self.Z), np.nanmax(self.Z)

                Mask2 = np.isfinite(self.Z)
                self.Z[~Mask2] = self.maxz

                # remove offset at center
                self.Z -= misc.ValueAt(self.xy, self.xy, self.Z, 0, 0)

            case "Function":
                self.minz = self.pos[2] + self.func.minz
                self.maxz = self.pos[2] + self.func.maxz
                self.r = self.func.r
                self.eps = max(self.r / 1e6, 1e-12)

            case "Rectangle":
                self.minz = self.maxz = self.pos[2]
                self.eps = max(self.dim[0] / 1e6, 1e-12)

            case ("Circle" | "Line"):
                self.minz = self.maxz = self.pos[2]
           
            case "Ring":
                self.minz = self.maxz = self.pos[2]
                if ri >= r:
                    raise ValueError("ri needs to be smaller than r.")

            case "Point":
                self.minz = self.maxz = self.pos[2]
                self.r = 0
                self.eps = 0

            case _:
                raise ValueError("Invalid surface_type")

    def copy(self) -> 'Surface':
        """

        :return:
        """
        return copy.deepcopy(self)

    def hasHitFinding(self) -> bool:
        """

        :return:
        """
        return self.surface_type in ["Circle", "Ring", "Rectangle", "Sphere", "Asphere"] \
                or (self.surface_type == "Function" and self.func.hasHits())

    def isPlanar(self) -> bool:
        """

        :return:
        """
        return self.surface_type in ["Circle", "Ring", "Rectangle", "Point", "Line"]
    
    def moveTo(self, pos: (list | np.ndarray)) -> None:
        """
        Moves the surface in 3D space.

        :param pos: 3D position to move to (list or numpy 1D array)
        """

        self.minz += pos[2] - self.pos[2]
        self.maxz += pos[2] - self.pos[2]

        # update position
        self.pos = np.array(pos, dtype=np.float64)
    
    def getExtent(self) -> tuple[float, float, float, float, float, float]:
        """

        :return:
        """
        match self.surface_type:

            case ("Circle" | "Ring" | "Sphere" | "Asphere" | "Function" | "Data" | "Point"):
                return *(self.r*np.array([-1, 1, -1, 1]) + self.pos[:2].repeat(2)), \
                       self.minz, self.maxz

            case "Rectangle":
                return *(self.pos[:2].repeat(2) + self.dim.repeat(2)/2 * np.array([-1, 1, -1, 1])), \
                        self.minz, self.maxz

            case "Line":
                return self.pos[0] - self.r*np.cos(self.ang),\
                       self.pos[0] + self.r*np.cos(self.ang),\
                       self.pos[1] - self.r*np.sin(self.ang),\
                       self.pos[1] + self.r*np.sin(self.ang),\
                       self.minz,\
                       self.maxz
            case _:
                raise RuntimeError(f"No extent defined for surface_type {self.surface_type}.")
    
    def getValues(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface values.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        match self.surface_type:
            case ("Asphere" | "Sphere"):

                x0, y0, z0 = self.pos
                r2 = misc.calc("(x-x0)**2 + (y-y0)**2")

                inside = r2 <= self.r**2

                z = np.full_like(r2, self.maxz, dtype=np.float64)

                rho, k = self.rho, self.k
                z[inside] = misc.calc("z0 + rho*r2i/(1 + sqrt(1 - (k+1)* rho**2 *r2i))", r2i=r2[inside])

                return z

            case "Function":
                return self.pos[2] + self.func.getValues(x-self.pos[0], y-self.pos[1])

            case "Data":
                xs, xe, ys, ye, _, _ = self.getExtent()
                inside = (xs < x) & (x < xe) & (ys < y) & (y < ye)

                z = np.full(x.shape, self.maxz, dtype=np.float64)

                if np.count_nonzero(inside):
                    z[inside] = self.pos[2] + misc.interp2d(self.xy, self.xy, self.Z, x[inside] - self.pos[0], y[inside] - self.pos[1])

                return z

            case ("Circle" | "Ring" | "Rectangle"):
                return np.full(x.shape, self.pos[2], dtype=np.float64)

            case _:
                raise RuntimeError(f"Surface value function not defined for surface_type {self.surface_type}")

    # TODO Kommentare
    # TODO Point and Line
    # TODO im Fall Blende Ringbereich formen in der Mitte
    def getPlottingMesh(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D plotting mesh. Note that the values are not gridded, the distance can be arbitrary.
        The only guarantee is that neighbouring array values are neighbouring values in 3D space.

        :param N: number of grid values in each dimension (int)
        :return: X, Y, Z coordinate array (all numpy 2D array)
        """
        if self.surface_type == "Rectangle":
            xs, xe, ys, ye, _, _ = self.getExtent()
            x = np.linspace(xs, xe, N)
            y = np.linspace(ys, ye, N)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(Y, self.pos[2], dtype=np.float64)
            return X, Y, Z

        else:
            x = self.pos[0] + np.linspace(-self.r, self.r, N)
            y = self.pos[1] + np.linspace(-self.r, self.r, N)

        X, Y = np.meshgrid(x, y)
        z = self.getValues(X.ravel(), Y.ravel())

        R = np.sqrt((X-self.pos[0])**2 + (Y-self.pos[1])**2)
        Phi = np.arctan2(Y-self.pos[1], X-self.pos[0])

        r = self.r - self.eps
        mask = R.ravel() >= r
        mask2 = R >= r

        # set values outside surface to surface edge
        z[mask] = self.getValues(self.pos[0] + r*np.cos(Phi.ravel()[mask]), self.pos[1] + r*np.sin(Phi.ravel()[mask]))
        X[mask2] = self.pos[0] + r*np.cos(Phi[mask2])
        Y[mask2] = self.pos[1] + r*np.sin(Phi[mask2])

        # extra precautions to plot the inner ring
        if self.surface_type == "Ring":
            # move values inside first 1/3 of surface ring slightly below ri
            # to plot the hole
            mask4 = (R > self.ri) & (R < (self.ri + (self.r-self.ri)*1/3))
            ris = self.ri - self.eps
            X[mask4] = self.pos[0] + ris*np.cos(Phi[mask4])
            Y[mask4] = self.pos[1] + ris*np.sin(Phi[mask4])

            # move values inside second 1/3 of ring slightly above ri
            # to plot ring edge
            mask5 = ~mask4 & (R < (self.ri + (self.r-self.ri)*2/3))
            rie = self.ri+self.eps
            X[mask5] = self.pos[0] + rie*np.cos(Phi[mask5])
            Y[mask5] = self.pos[1] + rie*np.sin(Phi[mask5])

        # plot nan values inside
        mask3 = self.getMask(X.ravel(), Y.ravel())
        z[~mask & ~mask3] = np.nan

        Z = z.reshape((y.shape[0], x.shape[0]))

        return X, Y, Z

    def getMask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get surface mask values. A value of 1 means the surface is defined here.

        :param x: x-coordinate array (numpy 1D or 2D array)
        :param y: y-coordinate array (numpy 1D or 2D array)
        :return: z-coordinate array (numpy 1D or 2D array, depending on shape of x and y)
        """
        match self.surface_type:
            case "Data":
                xs, xe, ys, ye, _, _ = self.getExtent()
                inside = (xs < x) & (x < xe) & (ys < y) & (y < ye)

                Mask = np.array(self.Mask, dtype=np.float32)  # convert bool mask to float

                zv = np.zeros(x.shape)
                if np.count_nonzero(inside):
                    zv[inside] = misc.interp2d(self.xy, self.xy, Mask, x[inside] - self.pos[0], y[inside] - self.pos[1])

                return zv == 1

            case "Rectangle":
                xs, xe, ys, ye, _, _ = self.getExtent()
                inside = (xs < x) & (x < xe) & (ys < y) & (y < ye)
                return inside

            case "Function":
                return self.func.getMask(x-self.pos[0], y-self.pos[1])

            case ("Circle" | "Ring" | "Sphere" | "Asphere"):
                # use r^2 instead of r, saves sqrt calculation for all points
                r2 = misc.calc("(x - x0) ** 2 + (y - y0) ** 2", x0=self.pos[0], y0=self.pos[1])

                if self.surface_type == "Ring":
                    return (self.ri**2 < r2) & (r2 < self.r**2)
                else:
                    return r2 < self.r**2
            case _:
                raise RuntimeError(f"Mask function not defined for surface_type {self.surface_type}.")

    # TODO funktioniert das für konkave Flächen? woher minus in n_r
    def getNormals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get normal vectors of the surface.

        :param x: x-coordinates (numpy 1D array)
        :param y: y-coordinates (numpy 1D array)
        :return: normal vector array of shape (x.shape[0], 3), components in second dimension (numpy 2D array)
        """
        match self.surface_type:

            case ("Circle" | "Ring" | "Rectangle"):
                n = np.zeros((x.shape[0], 3), dtype=np.float64)
                n[:, 2] = 1
                return n

            case "Function" if self.func.hasDerivative():
                n = np.full((x.shape[0], 3), -1, dtype=np.float64, order='F')

                n[:, 0], n[:, 1] = self.func.getDerivative(x-self.pos[0], y-self.pos[1])

                misc.normalize(n)

                return -n

            case ("Sphere" | "Asphere"):

                x0, y0 = self.pos[:2]
                r = misc.calc("sqrt((x-x0)**2 + (y-y0)**2)")
                phi = misc.calc("arctan2(y, x)")

                # the derivative of a conic section formula is
                #       m := dz/dr = rho*r/sqrt(1 - (k+1)*rho**2*r**2)
                # the angle beta of the n_r-n_z normal triangle is
                #       beta = pi/2 + arctan(m)
                # the radial component n_r is
                #       n_r = cos(beta)
                # all this put together can be simplified to
                #       n_r = -rho*r / sqrt(1-k*r**2*rho**2)
                # since n = [n_r, n_z] is a unity vector
                #       n_z = +sqrt(1 - n_r**2)
                # this holds true since n_z is always positive in our raytracer

                rho, k = self.rho, self.k
                n_r = misc.calc("-rho*r/sqrt(1 - k*r**2*rho**2)")

                n = np.zeros((x.shape[0], 3), dtype=np.float64, order='F')

                # n_r is rotated by phi to get n_x, n_y
                misc.calc("n_r*cos(phi)",     out=n[:, 0])
                misc.calc("n_r*sin(phi)",     out=n[:, 1])
                misc.calc("sqrt(1 - n_r**2)", out=n[:, 2])

                return n
           
            # get normals the numerical way
            case _:
                ds = self.eps

                # uz is the surface change in x direction, vz in y-direction, the normal vector 
                # of derivative vectors u = (ds, 0, uz) and v = (0, ds, vz) is n = (-uz*ds, -ds*vz, ds*ds)
                # this can be rescaled to n = (-uz, -vz, ds)
                n = np.zeros((x.shape[0], 3), dtype=np.float64, order='F')
                n[:, 0] = self.getValues(x - ds/2, y) - self.getValues(x + ds/2, y) # -uz
                n[:, 1] = self.getValues(x, y - ds/2) - self.getValues(x, y + ds/2) # -vz
                n[:, 2] = ds

                # normalize vectors
                misc.normalize(n)

                return n

    # TODO rectangular edge
    # TODO use case rectangular edge for surface_type = Data?
    def getEdge(self, nc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface values of the surface edge, assumes a circular edge.

        :param nc: number of points on edge (int)
        :return: X, Y, Z coordinate arrays (all numpy 2D array)
        """
        if self.surface_type ==  "Rectangle":
            N4 = int(nc/4)
            dn = nc - 4*N4
            xs, xe, ys, ye, _, _ = self.getExtent()

            x = np.concatenate((np.linspace(xs, xe, N4), 
                                np.full((N4,), xe), 
                                np.flip(np.linspace(xs, xe, N4)), 
                                np.full((N4+dn, ), xs)))

            y = np.concatenate((np.full((N4,), ys), 
                                np.linspace(ys, ye, N4), 
                                np.full((N4, ), ye),
                                np.flip(np.linspace(ys, ye, N4+dn))))

            return x, y, np.full_like(y, self.pos[2])

        else:
            r = self.r - self.eps

            theta = np.linspace(0, 2 * np.pi, nc)
            xd = self.pos[0] + r * np.cos(theta)
            yd = self.pos[1] + r * np.sin(theta)
            zd = self.getValues(xd, yd)

            return xd, yd, zd

    def getRandomPositions(self, N: int) -> np.ndarray:
        """

        :param N:
        :return:
        """

        p = np.zeros((N, 3), dtype=np.float64, order='F')
        
        match self.surface_type:

            case ("Circle" | "Ring"):
                rs = 0 if self.surface_type == "Circle" else self.ri  # choose inner radius
                # weight with square root to get equally distributed points
                r0 = np.sqrt(np.random.uniform((rs/self.r)**2, 1, N))
                theta = np.random.uniform(0, 2*np.pi, N)

                x0, y0, r = self.pos[0], self.pos[1], self.r

                misc.calc("x0 + r0*r*cos(theta)", out=p[:, 0])
                misc.calc("y0 + r0*r*sin(theta)", out=p[:, 1])
                p[:, 2] = self.pos[2]

            case "Rectangle":
                ext = self.getExtent()
                p[:, 0] = np.random.uniform(ext[0], ext[1], N)
                p[:, 1] = np.random.uniform(ext[2], ext[3], N)
                p[:, 2] = self.pos[2]

            case "Line":
                t = np.random.uniform(-self.r, self.r, N)
                ang = self.ang / 180 * np.pi
                p[:, 0] = self.pos[0] + np.cos(ang)*t
                p[:, 1] = self.pos[1] + np.sin(ang)*t
                p[:, 2] = self.pos[2]

            case "Point":
                p[:, 0] = self.pos[0]
                p[:, 1] = self.pos[1]
                p[:, 2] = self.pos[2]

            case _:
                raise RuntimeError(f"surface_type '{self.surface_type}' not handled.")

        return p
        

    # TODO does this work for non-hitting rays?
    # TODO check with  k = -1 and sz = 1
    def findHit(self, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param p:
        :param s:
        :return:
        """
        match self.surface_type:

            case ("Circle" | "Ring" | "Rectangle"):
                t = (self.pos[2] - p[:, 2])/s[:, 2]
                p_hit = p + s*t[:, np.newaxis]
                is_hit = self.getMask(p_hit[:, 0], p_hit[:, 1])
                return p_hit, is_hit

            case "Function" if self.func.hasHits():

                p_hit0 = self.func.getHits(p- self.pos, s)
                is_hit = self.func.getMask(p_hit0[:, 0], p_hit0[:, 1])

                return p_hit0 + self.pos, is_hit

            case ("Sphere" | "Asphere"):

                ox = p[:, 0] - self.pos[0]
                oy = p[:, 1] - self.pos[1]
                oz = p[:, 2] - self.pos[2]

                sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
                k, rho = self.k, self.rho

                A = misc.calc("1 + k*sz**2")
                B = misc.calc("sx*ox + sy*oy + sz*(oz*(k+1) - 1/rho)")
                C = misc.calc("oy**2 + ox**2 + oz*(oz*(k+1) - 2/rho)")

                # if there are not hits, this term gets imaginary
                # we'll handle this case afterwards
                D = misc.calc("sqrt(B**2 - C*A)")

                # we get two possible ray parameters
                # nan values for A == 0, we'll handle them later
                t1 = misc.calc("(-B - D) / A")
                t2 = misc.calc("(-B + D) / A")

                # choose t that leads to z-position inside z-range of surface
                z1 = p[:, 2] + sz*t1
                minz, maxz = self.minz, self.maxz
                t = misc.calc("where((minz <= z1) & (z1 <= maxz), t1, t2)")

                # calculate hit points and hit mask
                p_hit = p + s*t[:, np.newaxis]
                is_hit = self.getMask(p_hit[:, 0], p_hit[:, 1])

                # case A == 0 and B != 0 => one intersection
                mask = (A == 0) & (B != 0)
                if np.any(mask):
                    t = -C[mask]/(2*B[mask])
                    p_hit[mask] = p[mask] + s[mask]*t[:, np.newaxis]
                    is_hit[mask] = self.getMask(p_hit[mask, 0], p_hit[mask, 1])

                # no intersection, set intesection to imaginary plane z = maxz
                # no hit when ~is_hit, D imaginary, (A == 0 and B == 0), or hit point behind starting point (-z direction)
                nh = ~is_hit | ~np.isfinite(D) | ((A == 0) & (B == 0)) | (p_hit[:, 2] < p[:, 2])
                tnh = (self.maxz - p[nh, 2])/s[nh, 2]
                p_hit[nh] = p[nh] + s[nh]*tnh[:, np.newaxis]
                is_hit[nh] = False

                return p_hit, is_hit

            case _:
                raise RuntimeError(f"Hit finding not defined for surface_type {self.surface_type}.")

