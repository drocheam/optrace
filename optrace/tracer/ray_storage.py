
import numpy as np  # calculations

from .geometry import RaySource  # create rays
from .base_class import BaseClass  # parent class
from . import misc  # calculations


class RayStorage(BaseClass):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self._lock = False
        self.N_list = np.array([], dtype=int)
        self.B_list = np.array([], dtype=int)
        
        self.no_pol = False
        self.ray_source_list = []

        self.p_list = np.array([])
        self.s0_list = np.array([])
        self.n_list = np.array([])
        self.pol_list = np.array([])
        self.w_list = np.array([])
        self.wl_list = np.array([])

        super().__init__(**kwargs)

    def init(self,
             ray_source_list:   list[RaySource],
             N:                 int,
             nt:                int,
             no_pol:            bool)\
            -> None:
        """

        :param ray_source_list:
        :param N:
        :param nt:
        :param no_pol:
        :return:
        """
        self._lock = False
        self.no_pol = no_pol

        assert N >= 0
        assert nt >= 0
        assert len(ray_source_list)

        # all rays have the same starting power.
        # The rays are distributed between the sources according to the ray source power
        # get source powers and overall power
        P_list = np.array([RS.power for RS in ray_source_list])
        P_all = np.sum(P_list)

        # calculate int ray number from power ratio
        self.N_list = (N * P_list / P_all).astype(int)
        dN = N - np.sum(self.N_list)  # difference to ray number

        # distribute difference dN randomly on all sources, with the power being the probability
        index_add = np.random.choice(self.N_list.shape[0], size=dN, p=P_list / P_all)
        np.add.at(self.N_list, index_add, np.ones(index_add.shape))

        if np.any(np.array(self.N_list) == 0):
            self.print("There are RaySources that have no rays assigned. "
                       "Change the power ratio or raise the overall ray number")

        self.B_list = np.concatenate(([0], np.cumsum(self.N_list))).astype(int)
        self.ray_source_list = ray_source_list

        # save some storage space if we need the space only for nans when self.no_pol = True
        pol_dtype = np.float32 if not self.no_pol else np.float16

        # weights, polarization and wavelengths don't need double precision, this way we save some RAM
        # fortran order='F' speeds things up around 20% in our application
        self.p_list = np.zeros((N, nt, 3), dtype=np.float64, order='F')
        self.s0_list = np.zeros((N, 3), dtype=np.float64, order='F')
        self.pol_list = np.zeros((N, nt, 3), dtype=pol_dtype, order='F')
        self.w_list = np.zeros((N, nt), dtype=np.float32, order='F')
        self.n_list = np.zeros((N, nt), dtype=np.float64, order='F')
        self.wl_list = np.zeros(N, dtype=np.float32)

    @property
    def N(self) -> int:
        """number of rays"""
        return self.p_list.shape[0] if self.N_list.shape[0] else 0

    @property
    def Nt(self) -> int:
        """number of ray sections per ray"""
        return self.p_list.shape[1] if self.N_list.shape[0] else 0

    def make_thread_rays(self, N_threads: int, Nt: int) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param N_threads:
        :param Nt:
        :return:
        """

        assert self.N, "ray_source_list has no rays stored."
        assert 0 <= Nt < N_threads

        Np = int(self.N/N_threads)  # rays per thread
        Ns = Nt*Np
        Ne = Ns + Np if Nt != N_threads-1 else self.N  # last threads also gets remainder

        i = np.argmax(self.B_list > Ns) - 1
        i = max(i, 0)  # enforce i >= 0

        while self.B_list[i] < Ne:

            Nsi = max(Ns, self.B_list[i])
            Nei = min(self.B_list[i + 1], Ne)
            sl1 = slice(Nsi, Nei)

            power = (Nei - Nsi) / self.N_list[i] * self.ray_source_list[i].power

            self.p_list[sl1, 0], self.s0_list[sl1], self.pol_list[sl1, 0], self.w_list[sl1, 0], self.wl_list[sl1]\
                = self.ray_source_list[i].create_rays(Nei - Nsi, no_pol=self.no_pol, power=power)

            i += 1

        return self.p_list[Ns:Ne], self.s0_list[Ns:Ne], self.pol_list[Ns:Ne],  self.w_list[Ns:Ne], self.wl_list[Ns:Ne], self.n_list[Ns:Ne]

    def source_sections(self, index: int = None)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param index:
        :return:
        """
        assert self.N, "ray_source_list has no rays stored."
        assert index is None or 0 <= index < len(self.N_list)

        Ns, Ne = self.B_list[index:index + 2] if index is not None else (0, self.N)

        return self.p_list[Ns:Ne, 0], self.s0_list[Ns:Ne], self.pol_list[Ns:Ne, 0],\
            self.w_list[Ns:Ne, 0], self.wl_list[Ns:Ne]

    def ray_lengths(self, ch: np.ndarray = None, ch2: np.ndarray = None) -> np.ndarray:
        """
        Euclidean lengths of ray sections.

        :param ch:
        :param ch2:
        :return:
        """
        _, s, _, _, _, _, _ = self.rays_by_mask(ch, ch2, ret=[0, 1, 0, 0, 0, 0, 0], normalize=False)
        return np.linalg.norm(s, axis=s.ndim-1)
    
    def optical_lengths(self, ch: np.ndarray = None) -> np.ndarray:
        """
        
        :param ch: boolean array for ray selection
        :return: Optical path length for each ray section
        """

        l = self.ray_lengths(ch)
        _, _, _, _, _, _, n = self.rays_by_mask(ch, ret=[0, 0, 0, 0, 0, 0, 1])
        return l*n

    def rays_by_mask(self,
                     ch:           np.ndarray = None,
                     ch2:          np.ndarray = None,
                     ret:          list[bool | int] = None,
                     normalize:    bool = True) \
            -> tuple[(np.ndarray | None), (np.ndarray | None), (np.ndarray | None),
                     (np.ndarray | None), (np.ndarray | None), (np.ndarray | None), (np.ndarray | None)]:
        """

        :param ch: bool array
        :param ch2:
        :param ret:
        :param normalize:
        :return:
        """
        assert self.N, "ray_source_list has no rays stored."
        
        # assign default parameter for ret
        ret = [1, 1, 1, 1, 1, 1, 1] if ret is None else ret

        ch = np.ones(self.N, dtype=bool) if ch is None else ch
        ch2 = slice(None) if ch2 is None else ch2
      
        assert ch.shape[0] == self.N

        # calculate source numbers
        if ret[5]:
            ind = np.nonzero(ch)[0]
            snums = np.zeros_like(ind, dtype=int)
            for i, _ in enumerate(self.N_list):
                Ns, Ne = self.B_list[i:i + 2]
                snums[(Ns <= ind) & (ind < Ne)] = i

        # calculate s
        if ret[1]:
            if not isinstance(ch2, slice):
                ch21 = np.where(ch2 < self.Nt - 1, ch2 + 1, ch2)
                s = self.p_list[ch, ch21] - self.p_list[ch, ch2]
                if normalize:
                    s = misc.normalize(s)
            else:
                s = self.p_list[ch, 1:] - self.p_list[ch, :-1]
                s = np.hstack((s, np.zeros((s.shape[0], 1, 3), order='F')))
                if normalize:
                    s_ = s.reshape((s.shape[0]*s.shape[1], 3))
                    s = misc.normalize(s_).reshape(s.shape)

        return self.p_list[ch, ch2] if ret[0] else None,\
            s if ret[1] else None,\
            self.pol_list[ch, ch2] if ret[2] else None,\
            self.w_list[ch, ch2] if ret[3] else None,\
            self.wl_list[ch] if ret[4] else None,\
            snums if ret[5] else None,\
            self.n_list[ch] if ret[6] else None
