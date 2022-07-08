
from threading import Thread  # threading
import warnings  # print warnings
import numpy as np  # calculations

from optrace.tracer.geometry.RaySource import RaySource  # create Rays
import optrace.tracer.Misc as misc # calculations
from optrace.tracer.BaseClass import BaseClass  # parent class


class RayStorage(BaseClass):

    N_list = np.array([], dtype=int)
    B_list = np.array([], dtype=int)
  
    def init(self, RaySourceList, N, nt, no_pol=False):
        """
        """
        self._lock = False
        self.no_pol = no_pol

        # all rays have the same starting power.
        # The rays are distributed between the sources according to the ray source power
        # get source powers and overall power
        P_list = np.array([RS.power for RS in RaySourceList])
        P_all = np.sum(P_list)

        # calculate int ray number from power ratio
        self.N_list = (N * P_list / P_all).astype(int)
        dN = N - np.sum(self.N_list)  # difference to ray number

        # distribute difference dN randomly on all sources, with the power being the probability
        index_add = np.random.choice(self.N_list.shape[0], size=dN, p=P_list/P_all)
        np.add.at(self.N_list, index_add, np.ones(index_add.shape))

        if np.any(np.array(self.N_list) == 0) and not self.silent:
            warnings.warn("There are RaySources that have no rays assigned. "\
                    "Change the power ratio or raise the overall ray number", RuntimeWarning)
        
        self.B_list = np.concatenate(([0], np.cumsum(self.N_list))).astype(int)
        self.RaySourceList = RaySourceList

        # save some storage space if we need the space only for nans when self.no_pol = True
        pol_dtype = np.float32 if not self.no_pol else np.float16

        # weights, polarization and wavelengths don't need double precision, this way we save some RAM
        # fortran order='F' speeds things up around 20% in our application
        self.p_list      = np.zeros((N, nt, 3), dtype=np.float64, order='F')
        self.s0_list     = np.zeros((N, 3),     dtype=np.float64, order='F')
        self.pol_list    = np.zeros((N, nt, 3), dtype=pol_dtype,  order='F')
        self.w_list      = np.zeros((N, nt),    dtype=np.float32, order='F')
        self.wl_list     = np.zeros(N,          dtype=np.float32)     

    @property
    def N(self) -> int:
        """number of rays"""
        return self.p_list.shape[0] if self.N_list.shape[0] else 0

    @property
    def nt(self) -> int:
        """number of ray sections"""
        return self.p_list.shape[1] if self.N_list.shape[0] else 0
    
    def makeThreadRays(self, N_threads: int, Nt: int) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param N_threads:
        :param Nt:
        :return:
        """

        if not self.N:
            raise RuntimeError("RaySourceList has no rays stored.")
        
        Np = int(self.N/N_threads) # rays per thread
        Ns = Nt*Np
        Ne = Ns + Np if Nt != N_threads-1 else self.N  # last threads also gets remainder

        i = np.argmax(self.B_list > Ns) - 1
        i = max(i, 0)  # enforce i >= 0

        while self.B_list[i] < Ne:
      
            Nsi = max(Ns, self.B_list[i])
            Nei = min(self.B_list[i+1], Ne)
            sl1 = slice(Nsi, Nei)

            power = (Nei - Nsi) / self.N_list[i] * self.RaySourceList[i].power

            self.p_list[sl1, 0], self.s0_list[sl1], self.pol_list[sl1, 0], self.w_list[sl1, 0], self.wl_list[sl1]\
                         = self.RaySourceList[i].createRays(Nei-Nsi, no_pol=self.no_pol, power=power)

            i += 1

        return self.p_list[Ns:Ne], self.s0_list[Ns:Ne], self.pol_list[Ns:Ne],  self.w_list[Ns:Ne], self.wl_list[Ns:Ne]

    def getSourceSections(self, index: int | None)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param index:
        :return:
        """
        if not self.N:
            raise RuntimeError("RaySourceList has no rays stored.")
        
        Ns, Ne = self.B_list[index:index+2] if index is not None else (0, self.N)

        return self.p_list[Ns:Ne, 0], self.s0_list[Ns:Ne], self.pol_list[Ns:Ne, 0],\
               self.w_list[Ns:Ne, 0], self.wl_list[Ns:Ne]

    def getRaysByMask(self,
                      ch:           np.ndarray,
                      ch2:          np.ndarray = None,
                      ret:          list[bool | int] = [1, 1, 1, 1, 1, 1]) \
            -> tuple[(np.ndarray | None), (np.ndarray | None), (np.ndarray | None),
                     (np.ndarray | None), (np.ndarray | None), (np.ndarray | None)]:
        """

        :param ch: bool array
        :param ch2:
        :param ret:
        :return:
        """

        if not self.N:
            raise RuntimeError("RaySourceList has no rays stored.")
       
        ch2 = slice(None) if ch2 is None else ch2

        # calculate source numbers
        if ret[5]:
            ind = np.nonzero(ch)[0]
            snums = np.zeros_like(ind, dtype=int)
            for i, _ in enumerate(self.N_list):
                Ns, Ne = self.B_list[i:i+2]
                snums[(Ns <= ind) & (ind < Ne)] = i

        # calculate s
        if ret[1]:
            s = self.p_list[ch, 1:] - self.p_list[ch, :-1]
            s = np.hstack((s, s[:, np.newaxis, -1]))

            if not isinstance(ch2, slice):
                s = s[np.ones(s.shape[0], dtype=bool), ch2]
                s = misc.normalize(s)
            else:
                s_ = s.reshape((s.shape[0]*s.shape[1], 3))
                s_ = misc.normalize(s_)
                s = s_.reshape(s.shape)

        p     = self.p_list[ch, ch2]    if ret[0] else None
        s     = s                       if ret[1] else None
        pol   = self.pol_list[ch, ch2]  if ret[2] else None
        w     = self.w_list[ch, ch2]    if ret[3] else None
        wl    = self.wl_list[ch]        if ret[4] else None
        snums = snums                   if ret[5] else None

        return p, s, pol, w, wl, snums

