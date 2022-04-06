
from threading import Thread

import numpy as np
from Backend.RaySource import *
import Backend.Misc as misc


class RayStorage:

    N_list = np.array([])
    B_list = np.array([])

    def hasRays(self) -> bool:
        """

        :return:
        """
        return len(self.N_list) > 0

    @property
    def N(self) -> int:
        """number of rays"""
        return self.p_list.shape[0] if self.hasRays() else None

    @property
    def nt(self) -> int:
        """number of ray sections"""
        return self.p_list.shape[1] if self.hasRays() else None

    def createRays(self,
                   RaySourceList:   list[RaySource],
                   N_list:          list[int],
                   nt:              int,
                   threading:       bool = False,
                   no_pol:          bool = False) \
            -> None:
        """

        :param RaySourceList:
        :param N_List:
        :param nt:
        :param threading:
        :param no_pol:
        """
        self._locked = False
        self.N_list = np.array(N_list, dtype=int)
        self.B_list = np.concatenate(([0], np.cumsum(self.N_list)))
        N           = np.sum(N_list)
       

        # weights, polarization and wavelengths don't need double precision, this way we save some RAM
        # fortran order='F' speeds things up around 20% in our application
        self.p_list      = np.zeros((N, nt, 3), dtype=np.float64, order='F')
        self.s0_list     = np.zeros((N, 3),     dtype=np.float64, order='F')
        self.pol_list    = np.zeros((N, nt, 3), dtype=np.float32, order='F')
        self.w_list      = np.zeros((N, nt),    dtype=np.float32, order='F')
        self.wl_list     = np.zeros((N,),       dtype=np.float32)        

        def addSourceRays(i: int) -> None:

            Ns, Ne = self.B_list[i:i+2]
            sl1 = slice(Ns, Ne)

            self.p_list[sl1, 0],\
            self.s0_list[sl1],   \
            self.pol_list[sl1, 0],\
            self.w_list[sl1, 0],   \
            self.wl_list[sl1]   \
                                     = RaySourceList[i].createRays(N_list[i], no_pol=no_pol)

        # don't use multithreading if there are to many ray sources
        if threading and len(RaySourceList) < 2*misc.getCoreCount():

            thread_list = [Thread(target=addSourceRays, args=[N_t]) for N_t in np.arange(len(RaySourceList))]
            [thread.start() for thread in thread_list]
            [thread.join()  for thread in thread_list]

        else:
            for i in np.arange(len(RaySourceList)):
                addSourceRays(i)

    def getThreadRays(self, N_threads: int, Nt: int) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param N_threads:
        :param Nt:
        :return:
        """
        # get subset of rays for multithreading
        
        if not self.hasRays():
            raise RuntimeError("RaySourceList has no rays stored.")

        Np = int(self.N/N_threads) # rays per thread
        Ns = Nt*Np
        Ne = Ns + Np if Nt != N_threads-1 else self.N  # last threads also gets remainder

        return self.p_list[Ns:Ne], \
               self.s0_list[Ns:Ne], \
               self.pol_list[Ns:Ne], \
               self.w_list[Ns:Ne],    \
               self.wl_list[Ns:Ne]

    def getRaysAtZ(self,
                   z:           float,
                   choice:      np.ndarray = [],
                   normalize:   bool = True)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get x, y coordinates of rays at position z. Only rays with weights are returned.

        returns:
        pz array: numpy 2D array, rays with weights in first dimension, xyz coordinates in second
        p array, s array (ubsboth numpy 2D array)
        weights array (numpy 1D array)

        :param z: z-position (float)
        :param choice: index array for rays (numpy 1D array)
        :param normalize:
        :return: pz array, p array, s array, weights array
        """

        if not self.hasRays():
            raise RuntimeError("RaySourceList has no rays stored.")
        
        N = self.N

        if choice != []:
            rays_pos = np.zeros((N,), dtype=bool)
            rays_pos[choice] = True
            pos = np.argmax(z < self.p_list[rays_pos, :, 2], axis=1) - 1
        else:
            pos = np.argmax(z < self.p_list[:, :, 2], axis=1) - 1
            rays_pos = np.ones((N, ), dtype=bool)


        # when z lies before the RaySource, pos gets set to -1. 
        # Since the last surface absorbs all rays, the weight is set to 0 correctly
        # and we don't need to do that

        # get Ray parts
        p, s, pol, w, wl, snum = self.getRaysByMask(rays_pos, pos, normalize=normalize)

        # get positions of rays at the detector
        hw = w > 0
        t = (z - p[hw, 2])/s[hw, 2]
        ph = p[hw] + s[hw] * t[:, np.newaxis]

        return ph, p, s, pol, w, wl, snum

    # TODO use last known s instead nan
    def getRay(self, num: int, normalize: bool = True) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
        """

        :param num:
        :param normalize:
        :return:
        """

        if not self.hasRays():
            raise RuntimeError("RaySourceList has no rays stored.")

        if num > self.N:
            raise ValueError("Number exceeds number of rays")

        snum = -1
        for i in np.arange(len(self.N_list)):
            Ns, Ne = self.B_list[i:i+2]
            if Ns <= num < Ne:
                snum = i 

        s = self.p_list[num, 1:] - self.p_list[num, :-1]
        s = np.concatenate((s, [s[-1]]))

        if normalize:
            norms = np.linalg.norm(s, axis=1)
            s[norms == 0] = np.nan
            s[norms != 0] = s[norms != 0] / norms[norms != 0, np.newaxis]

        return self.p_list[num], s, self.pol_list[num], self.w_list[num], self.wl_list[num], snum+1

    def getSourceRays(self, index: int)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param index:
        :return:
        """
        # get ray properties from source

        if not self.hasRays():
            raise RuntimeError("RaySourceList has no rays stored.")
        
        Ns, Ne = self.B_list[index:index+2] 

        return self.p_list[Ns:Ne, 0], \
               self.s0_list[Ns:Ne],    \
               self.pol_list[Ns:Ne, 0], \
               self.w_list[Ns:Ne, 0],    \
               self.wl_list[Ns:Ne]

    # TODO schöner machen
    # TODO was bei s = [0, 0, 0] und normalize=False?
    def getRaysByMask(self,
                      ch:           np.ndarray,
                      ch2:          np.ndarray = None,
                      normalize:    bool = True,
                      ret:          list[bool] = [True, True, True, True, True, True]) \
            -> tuple[(np.ndarray | None),
                     (np.ndarray | None),
                     (np.ndarray | None),
                     (np.ndarray | None),
                     (np.ndarray | None),
                     (np.ndarray | None)]:
        """

        :param ch:
        :param ch2:
        :param normalize:
        :param ret:
        :return:
        """
        # choice: bool array

        if not self.hasRays():
            raise RuntimeError("RaySourceList has no rays stored.")
        
        if not ret[5]:
            snums = None
        else:
            ind = np.nonzero(ch)[0]
            snums = np.zeros_like(ind, dtype=int)
            for i in np.arange(len(self.N_list)):
                Ns, Ne = self.B_list[i:i+2]
                snums[(ind >= Ns) & (ind < Ne)] = i
            snums += 1

        if not ret[1]:
            s = None
            if ch2 is None:
                ch2 = slice(None)
        else:
            if ch2 is None:
                ch2 = slice(None)
                s = self.p_list[ch, 1:] - self.p_list[ch, :-1]
                s = np.hstack((s, s[:, np.newaxis, -1]))

                if normalize:
                    norms = np.linalg.norm(s, axis=2)
                    mask = norms != 0
                    s[mask] /= norms[mask][:, np.newaxis]
                    s[~mask] = np.nan
            else:
                # init s vector
                s = np.zeros((ch2.shape[0], 3), dtype=np.float64)

                # rays outside mask have no s vector, since they are absorbed at the last outline area
                mask = ch2 + 1 < self.nt
                chm = np.zeros_like(ch, dtype=bool)
                chm[ch] = mask

                s[mask] = self.p_list[chm, ch2[mask]+1] - self.p_list[chm, ch2[mask]]

                if normalize:
                    misc.normalize(s)

        p   = self.p_list[ch, ch2]   if ret[0] else None
        pol = self.pol_list[ch, ch2] if ret[2] else None
        w   = self.w_list[ch, ch2]   if ret[3] else None
        wl  = self.wl_list[ch]       if ret[4] else None

        return p, s, pol, w, wl, snums

    def crepr(self):
        """

        """

        return [self.N, self.nt, tuple(self.N_list), tuple(self.B_list), 
                    id(self.p_list), id(self.s0_list), id(self.pol_list), id(self.w_list), id(self.wl_list)]
    

    # methods for signalising the user to not edit rays after raytracing
    # since it's python he could find a workaround for this, but maybe knowing there's maybe a reason for locking stops him doing so

    def lock(self):
        """make storage read only"""

        if self.p_list is not None:         self.p_list.flags.writeable = False
        if self.s0_list is not None:        self.s0_list.flags.writeable = False
        if self.w_list is not None:         self.w_list.flags.writeable = False
        if self.wl_list is not None:        self.wl_list.flags.writeable = False
        if self.pol_list is not None:       self.pol_list.flags.writeable = False

        self._locked = True

    def __setattr__(self, key, val):

        if "_locked" in self.__dict__ and self._locked and key != "_locked":
            raise RuntimeError("Operation not permitted since RayStorage is read-only outside raytracing.")

        self.__dict__[key] = val
