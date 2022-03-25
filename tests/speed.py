
import sys
sys.path.append('./src/')

from Backend import *

import copy
import time

start = time.time()

def func2(N2):
    # make Raytracer
    RT = Raytracer(outline=[-5, 5, -5, 5, 0, 60],  AbsorbMissing=True, silent=True)

    # add Raysource
    RSS = Surface("Circle", r=1)
    RS = RaySource(RSS, direction_type="Parallel", light_type="Blackbody",
                   pos=[0, 0, 0], s=[0, 0, 1], polarization_type='y')
    RT.add(RS)


    RS2 = RaySource(RSS, direction_type="Parallel", light_type="D65",
                   pos=[0, 1, 0], s=[0, 0, 1], polarization_type='x', power=2)
    RT.add(RS2)


    front = Surface(surface_type="Circle", r=3, rho=1/10, k=-0.444)
    back = Surface(surface_type="Circle", r=3, rho=-1/10, k=-7.25)
    nL2 = RefractionIndex("Constant", n=1.8)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 2], n=nL2)
    RT.add(L1)

    # add Lens 1
    front = Surface(surface_type="Asphere", r=3, rho=1/10, k=-0.444)
    back = Surface(surface_type="Asphere", r=3, rho=-1/10, k=-7.25)
    nL1 = RefractionIndex("Cauchy", A=1.49, B=0.00354)
    L1 = Lens(front, back, de=0.1, pos=[0, 0, 10], n=nL1)
    RT.add(L1)

    # add Lens 2
    front = Surface(surface_type="Asphere", r=3, rho=1/5, k=-0.31)
    back = Surface(surface_type="Asphere", r=3, rho=-1/5, k=-3.04)
    nL2 = RefractionIndex("Constant", n=1.8)
    L2 = Lens(front, back, de=0.6, pos=[0, 0, 25], n=nL2)
    RT.add(L2)

    # add Aperture
    ap = Surface(surface_type="Ring", r=1, ri=0.01)
    RT.add(Filter(ap, pos=[0, 0, 20.3]))

    # add Lens 3
    front = Surface(surface_type="Sphere", r=1, rho=1/2.2)
    back = Surface(surface_type="Sphere", r=1, rho=-1/5)
    nL3 = RefractionIndex("Function", func=lambda l: 1.8 - 0.007*(l - 380)/400)
    nL32 = RefractionIndex("Constant", n=1.1)
    L3 = Lens(front, back, de=0.1, pos=[0, 0, 47], n=nL3, n2=nL32)
    RT.add(L3)

    # # add Aperture2
    ap = Surface(surface_type="Circle", r=1, ri=0.005)

    def func(l):
        w = l.copy()
        w[l > 500] = 0
        w[l <= 500] = 1
        return w

    RT.add(Filter(ap, pos=[0, 0, 45.2], filter_type="Function", func=func))

    # add Detector
    Det = Detector(Surface(surface_type="Rectangle", dim=[3, 3]), pos=[0, 0, 60])
    RT.add(Det)

    RT.trace(N=N2)
    # Im = RT.DetectorImage(500, extent="auto")



N = 1000000

# Nt = 1 # multiprocessing.cpu_count() - 1
# threads = []
# for i in range(Nt):
    # process = Process(target=func2, args=[int(N/Nt)])
    # process.start()
    # threads.append(process)

# for process in threads:
    # process.join()
func2(N)
print(time.time()-start)

