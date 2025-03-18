import os.path  # working with paths
import numpy as np  # np.ndarray type
import chardet  # detection of text file encoding

# media
from .refraction_index import RefractionIndex

# needed geometries
from .geometry.group import Group
from .geometry import Lens, PointMarker, Detector, Aperture
from .geometry.surface import CircularSurface, ConicSurface, SphericalSurface,\
                              RingSurface, AsphericSurface, Surface, RectangularSurface

from .presets import spectral_lines  # lines for Abbe number checking
from ..global_options import global_options as go
from ..warnings import warning


_agf_modes = ["Schott", "Sellmeier1", "Herzberger", "Sellmeier2", "Conrady", "Sellmeier3", "Handbook of Optics 1",
              "Handbook of Optics 2", "Sellmeier4", "Extended", "Sellmeier5", "Extended2", "Extended3"]
"""List of ZEMAX agf material file formula modes. The position in the list corresponds to the number code."""


def _read_lines(path: str) -> list[str]:
    """
    reads lines of a file into a list of strings.
    File encoding is detected automatically.

    :param path: filepath to load
    :return: list of strings
    """

    # check if file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found/ is not a file.")

    # unfortunately we can't know if the file is a text file with a correct BOM
    # therefore a correct encoding detection using the first bytes is not given
    # use the chardet module to detect the encoding

    # open in binary mode to check for encoding
    with open(path, "rb") as f:
        resp = chardet.detect(f.read())
    
    # read using detected encoding
    with open(path, "r", encoding=resp["encoding"]) as f:
        return f.readlines()


def load_agf(path: str) -> dict:
    """
    Load an .agf material catalogue

    :param path: filepath
    :return: dictionary of refractive media, keys are names, values are RefractionIndex objects
    """

    lines = _read_lines(path)  # load text lines from file
    n_dict = {}
    skip = False

    # file format documentation
    #
    # see ZEMAX® Optical Design Program User's Manual, July 8, 2011
    # https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf (pages 599-600)
    # and 
    # https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass_user_manual.pdf

    # relevant lines
    # NM <material name> <formula mode number> <> <index at center wavelength> <abbe number> <> <>
    # CD <coefficient 1> <coefficient 2> ...
    # LD <wavelength range for formula lower bound> <wavelength range for formula upper bound>

    for lin in lines:

        # get name and mode
        if lin[:2] == "NM":
            skip = False
            linw = lin.split()
            name = linw[1]  # name of material

            # check if valid mode. According to documentation mode numbers go from 1-12
            ind = int(float(linw[2]))-1
            if ind < 0 or ind > len(_agf_modes) - 1:
                warning(f"{name}: Unknown index formula mode number {ind+1}, skipping.")
                skip = True  # skip further steps, since we don't know what to do with this material
                continue
            
            mode = _agf_modes[ind]  # index mode for our RefractionIndex class
            nc = float(linw[4])  # index at center wavelength
            V = float(linw[5])  # abbe number

        # get formula coefficients
        elif lin[:2] == "CD" and not skip:
            coeff0 = [float(a) for a in lin.split()[1:]]  # str to float list
            cnt = RefractionIndex.coeff_count[mode]  # number of needed coefficients
            coeff = coeff0[0:cnt]  # slice list so we have correct number of coefficients
            coeff = coeff + [0.]*(cnt-len(coeff))  # fill missing coefficients with zeros

        # get wavelength range
        # after this we can create and the check the refraction index object
        elif lin[:2] == "LD" and not skip:
            try:
                n = RefractionIndex(mode, coeff=coeff, desc=name)

                # wavelength bounds for formula
                linw = lin.split()[1:]
                wl0 = float(linw[0])*1000
                wl1 = float(linw[1])*1000

                # no spectral overlap for testing (e.g. infrared materials)
                if wl0 > spectral_lines.FdC[0] or wl1 < spectral_lines.FdC[2]:
                    warning(f"{name} wavelength range [{wl0}, {wl1}]nm does not overlap with "
                            f"testing wavelengths {spectral_lines.FdC}nm, skipping index and Abbe number checks.")
                
                else:
                    nc1 = n(spectral_lines.d)
                    V1 = n.abbe_number(spectral_lines.FdC)

                    # index deviation
                    if np.abs(nc1 - nc) > 1e-4:
                        warning(f"{name}: Index from file is {nc}, but calculated index is {nc1}. "
                                 "This can be due to different probe wavelengths.")

                    # abbe number deviation
                    elif np.abs(V1 - V) > 0.3:
                        warning(f"{name}: The Abbe number from file is {V}, but calculated is {V1}. "
                                "This can be due to different probe wavelengths.")

                # assign to material dict
                n_dict[name] = n

            # some exception occurred, e.g. index is < 1 somewhere
            except Exception as err:
                warning(f"Error for material {name}: " + str(err))

    return n_dict


def load_zmx(filename: str, n_dict: dict = None, no_marker: bool = False) -> Group:
    """
    Load a ZEMAX geometry from a .zmx into a Group.
    See the documentation on the limitations of the import.

    :param filename: filepath
    :param n_dict: dictionary of RefractiveIndex in the geometry
    :param no_marker: if there should be no marker created for the .zmx description
    :return: Group including the geometry from the .zmx
    """

    # see
    # ZEMAX® Optical Design Program User’s Guide Version 9.0, chapter 29 "THE ZMX FILE FORMAT"
    # https://pdfcoffee.com/zemaxmanual-pdf-free.html

    lines = _read_lines(filename)
    n_dict = n_dict or {}

    Surfaces, dds, n0, long_desc = _zmx_to_surface_dicts(lines, n_dict)

    G = _surface_dicts_to_geometry(Surfaces, dds, n0, long_desc, no_marker)

    return G


# make surface from surface dict
def _make_surface(surf: dict) -> AsphericSurface | ConicSurface | CircularSurface | SphericalSurface:
    """
    Create a Surface from a surface property dictionary

    :param surf: surface property dictionary from _zmx_to_surface_dicts
    :return: Surface object with correct class and properties
    """

    # STANDARD mode can be circle, sphere or conic
    if surf["stype"] == "STANDARD":
        if np.isfinite(surf["R"]):
            if "k" in surf and surf["k"]:
                return ConicSurface(r=surf["r"], R=surf["R"], desc=surf["desc"], k=surf["k"])
            else:
                return SphericalSurface(r=surf["r"], R=surf["R"], desc=surf["desc"])
        else:
            return CircularSurface(r=surf["r"], desc=surf["desc"])

    elif surf["stype"] == "EVENASPH":
        return AsphericSurface(r=surf["r"], desc=surf["desc"], R=surf["R"], k=surf["k"], coeff=surf["parm"])

    else:
        raise RuntimeError(f"Surface mode " + str(surf["stype"]) + " not supported yet.")


def _zmx_to_surface_dicts(lines:  list[str], 
                          n_dict: dict[RefractionIndex])\
        -> tuple[list, list, RefractionIndex | None, str]:
    """
    Process the file lines into a list of Surfaces and distances.

    :param lines: list of lines from the loaded file
    :param n_dict: dictionary of RefractionIndex
    :return: Surface dictionary list, distance list, ambient index n0 before first surface, zmx file description
    """
    Surfaces = []
    dds = []
    long_desc = ""
    n0 = None

    for i, l in enumerate(lines):  # never exits normally # pragma: no branch

        if l[:4] == "NAME":
            long_desc = l[5:-1]

        elif l[:4] == "UNIT":
            unit1 = l.split()[1]
            if unit1 != "MM":
                raise RuntimeError(f"Unsupported Unit {unit1}.")

        elif l[:4] == "MODE":
            mode = l.split()[1]
            if mode != "SEQ":
                raise RuntimeError(f"Unsupported Mode {mode}.")

        elif l[:4] == "SURF":
            break

    i += 1
    surf_i = 0
    while i < len(lines):
        
        parm = [0.]*10
        dd = 0
        surf = dict(stype="STANDARD", desc="", k=0, R=np.inf)

        while i + 1 < len(lines) and lines[i][:4] != "SURF":
        
            l = lines[i]
        
            # surface type
            if l[2:6] == "TYPE":
                surf["stype"] = l.split()[1]

            # diameter
            elif l[2:6] == "DIAM":
                surf["r"] = max(float(l.split()[1]), 1e-9)

            # conic constant
            elif l[2:6] == "CONI":
                surf["k"] = float(l.split()[1])

            # comment
            elif l[2:6] == "COMM":
                surf["desc"] = l[7:-1]
            
            # coating
            elif l[2:6] == "COAT":
                warning(f"Coatings are not supported. Ignoring coating '{l[7:-1]}'.")
            
            # aperture stop
            elif l[2:6] == "STOP":
                surf["STOP"] = True

            # curvature circle
            elif l[2:6] == "CURV":
                rho = float(l.split()[1])
                surf["R"] = 1/rho if rho else np.inf 
            
            elif l[2:6] == "DISZ":
                dd = float(l.split()[1])
                dd = max(dd, 3*Surface.N_EPS)  # distance needs to be above zero

            elif l[2:6] == "PARM":
                ind, val = l.split()[1:3]
                parm[int(float(ind))-1] = float(val)

            # glas material
            elif l[2:6] == "GLAS":

                material = l.split()[1]
                   
                nc, V = [float(a) for a in l.split()[4:6]] if len(l.split()) > 6 else [None, None]

                if material == "___BLANK":
                    # materials missing in n_dict can be approximated if n and V are provided 
                    # instead and name is __BLANK
                    surf["n"] = RefractionIndex("Abbe", n=nc, V=V)

                elif material not in n_dict.keys():

                    if nc is not None and V is not None and nc > 1 and V > 0:
                        surf["n"] = RefractionIndex("Abbe", n=nc, V=V)

                    else:
                        raise RuntimeError(f"Material {material} missing in n_dict parameter.")

                else:
                    surf["n"] = n_dict[material]

            i += 1

        # zeroth surface has a DIST of infinity -> definition of ambient index before lenses
        if surf_i == 0 and not np.isfinite(dd):
            n0 = surf["n"] if "n" in surf else RefractionIndex("Constant", n=1)

        else:
            # add surface to list
            surf["parm"] = parm
            Surfaces.append(surf)
            dds.append(dd)
        
        surf_i += 1
        i += 1

    return Surfaces, dds, n0, long_desc

def _surface_dicts_to_geometry(Surfaces:    list, 
                               dds:         list, 
                               n0:          RefractionIndex | None, 
                               long_desc:   str,
                               no_marker:   bool)\
        -> Group:
    """
    Convert the list of Surfaces dictionary and distances into a Group
    
    :param Surfaces: list of Surface dictionaries from _zmx_to_surface_dicts
    :param dds: list of distances in mm
    :param n0: ambient index before first surface
    :param long_desc: description of geometry from zmx files
    :param no_marker: if no description marker should be set
    :return: Group containing the geometry
    """
    G = Group(long_desc=long_desc, n0=n0)

    # find first surface with refraction index, all surfaces before that are irrelevant
    i = 0
    while i < len(Surfaces) and "n" not in Surfaces[i]:
        i += 1

    z = 0

    # calculate largest radius
    rmax = 0
    for s in Surfaces:
        if "r" in s and s["r"] > rmax:
            rmax = s["r"]

    # set largest radius to undefined radii
    # (radii are undefined, when the material reaches indefinitely in lateral direction)
    for s in Surfaces:
        if "r" not in s:
            s["r"] = rmax

    while i < len(Surfaces):

        if "n" not in Surfaces[i]:

            # last surface is a detector, but only if it has r > 0
            if i + 1 == len(Surfaces) and "r" in Surfaces[i]:
                # add square detector
                r = Surfaces[i]["r"]
                surf = RectangularSurface(dim=[2*r, 2*r])
                DET = Detector(surf, pos=[0, 0, z], desc=Surfaces[i]["desc"])
                G.add(DET)
            
            elif "STOP" in Surfaces[i]:
                surf = Surfaces[i]
                r = max(G.extent[1] - G.extent[0], G.extent[3] - G.extent[2]) / 2
                r = max(surf["r"] + 1, r)

                surf = RingSurface(ri=surf["r"], r=r)
                ap = Aperture(surf, pos=[0, 0, z], desc=Surfaces[i]["desc"])
                G.add(ap)

            z += dds[i]
            i += 1
            continue

        surf1 = _make_surface(Surfaces[i])
        surf2 = _make_surface(Surfaces[i+1])

        # NOTE
        # Surface 1 with index, Surface 2 without index -> Surface 1 and 2 form a lens,
        #  gap, next lens consists of surface 3 and 4
        #
        # Surface 1 with index, Surface 2 with index -> Surface 1 and 2 form a lens, next lens starts with surface 2
        # for that we offset the start position of the next lens by a small value.
        # The gap is filled with the medium of the last element.
        n2 = Surfaces[i]["n"] if "n" in Surfaces[i+1] else RefractionIndex("Constant", n=1)

        # create lens
        L = Lens(surf1, surf2, n=Surfaces[i]["n"], pos=[0, 0, z], d1=0, d2=dds[i], n2=n2, desc=Surfaces[i]["desc"])
        G.add(L)

        # see NOTE above
        if "n" in Surfaces[i+1]:
            z += dds[i] + 1e-7
            i += 1
        else:
            z += dds[i] + dds[i+1]
            i += 2

    # add an object marker if a desc is provided in the zmx file
    if G.long_desc != "" and not no_marker:
        ext = G.extent
        ym = np.mean(ext[2:4])
        zm = np.mean(ext[4:6])
        xm = ext[0] - 1.5

        G.add(PointMarker(G.long_desc, [xm, ym, zm], label_only=True))

    return G

