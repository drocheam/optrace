from optrace.tracer.refraction_index import RefractionIndex  # RefractionIndex class
import numpy as np  # ndarray creation


# Glasses
#######################################################################################################################


BAF10: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.5851495, 0.00926681282, 0.143559385,
                                                             0.0424489805, 1.08521269, 105.613573],
                                         desc="BAF10", long_desc="N_BAF10 (SCHOTT)")
"""N_BAF10 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=BAF10&page=SCHOTT"""


BAK1: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.12365662, 0.00644742752, 0.309276848,
                                                            0.0222284402, 0.881511957, 107.297751],
                                        desc="BAK1", long_desc="N-BAK1 (SCHOTT)")
"""N-BAK1 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=BAK1&page=SCHOTT"""


BASF64: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.65554268, 0.0104485644, 0.17131977,
                                                              0.0499394756, 1.33664448, 118.961472],
                                          desc="BASF64", long_desc="N-BASF64 (SCHOTT)")
"""N-BASF64 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-BaSF&page=N-BASF64"""


BK7: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.03961212, 0.00600069867, 0.231792344,
                                                           0.0200179144, 1.01046945, 103.560653],
                                       desc="BK7", long_desc="N-BK7 (SCHOTT)")
"""N-BK7 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT """


F2: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.39757037, 0.00995906143, 0.159201403,
                                                          0.0546931752, 1.2686543, 119.248346],
                                      desc="F2", long_desc="N-F2 (SCHOTT)")
"""N-F2 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-F&page=F2"""


FK51A: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.971247817, 0.00472301995, 0.216901417,
                                                             0.0153575612, 0.904651666, 168.68133],
                                         desc="FK51A", long_desc="N-FK51A (SCHOTT)")
"""N-FK51A (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=FK51A&page=SCHOTT"""


Fused_Silica: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.6961663, 0.0684043**2, 0.4079426,
                                                                    0.1162414**2, 0.8974794, 9.896161**2],
                                                desc="Fused_Silica", long_desc="Fused silica (fused quartz)")
"""Fused silica (fused quartz), see https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson"""


K5: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.08511833, 0.00661099503, 0.199562005,
                                                          0.024110866, 0.930511663, 111.982777],
                                      desc="K5", long_desc="N-K5 (SCHOTT)")
"""N-K5 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-K&page=N-K5"""


LAF2: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.80984227, 0.0101711622, 0.15729555,
                                                            0.0442431765, 1.0930037, 100.687748],
                                        desc="LAF2", long_desc="N-LAF2 (SCHOTT)")
"""N-LAF2 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaF&page=N-LAF2"""


LAK8: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.33183167, 0.00620023871, 0.546623206,
                                                            0.0216465439, 1.19084015, 82.5827736],
                                        desc="LAK8", long_desc="N-LAK8 (SCHOTT)")
"""N-LAK8 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaK&page=N-LAK8"""


LAK22: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.14229781, 0.00585778594, 0.535138441,
                                                             0.0198546147, 1.04088385, 100.834017],
                                         desc="LAK22", long_desc="N-LAK22 (SCHOTT)")
"""N-LAK22 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaK&page=N-LAK22"""


LASF9: RefractionIndex = RefractionIndex("Sellmeier", coeff=[2.00029547, 0.0121426017, 0.298926886,
                                                             0.0538736236, 1.80691843, 156.530829],
                                         desc="LASF9", long_desc="N-LASF9 (SCHOTT)")
"""N-LASF9 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=LASF9&page=SCHOTT"""


LASF44: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.78897105, 0.00872506277, 0.38675867,
                                                              0.0308085023, 1.30506243, 92.7743824],
                                          desc="LASF44", long_desc="N-LASF44 (SCHOTT)")
"""N-LASF44 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-LaSF&page=N-LASF44"""


LF5: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.28035628, 0.00929854416, 0.163505973,
                                                           0.0449135769, 0.893930112, 110.493685],
                                       desc="LF5", long_desc="N-LF5 (SCHOTT)")
"""N-LF5 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-LF&page=LF5"""


SF5: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.52481889, 0.011254756, 0.187085527,
                                                           0.0588995392, 1.42729015, 129.141675],
                                       desc="SF5", long_desc="N-SF5 (SCHOTT)")
"""N-SF5 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SF5&page=SCHOTT"""


SF6: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.72448482, 0.0134871947, 0.390104889,
                                                           0.0569318095, 1.04572858, 118.557185],
                                       desc="SF6", long_desc="N-SF6 (SCHOTT)")
"""N-SF6 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-SF&page=SF6"""


SF10: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.62153902, 0.0122241457, 0.256287842,
                                                            0.0595736775, 1.64447552, 147.468793],
                                        desc="SF10", long_desc="N-SF10 (SCHOTT)")
"""N-SF10 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SF10&page=SCHOTT"""


SF11: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.73759695, 0.013188707, 0.313747346,
                                                            0.0623068142, 1.89878101, 155.23629],
                                        desc="SF11", long_desc="N-SF11 (SCHOTT)")
"""N-SF11 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SF11&page=SCHOTT"""


SF66: RefractionIndex = RefractionIndex("Sellmeier", coeff=[2.0245976, 0.0147053225, 0.470187196,
                                                            0.0692998276, 2.59970433, 161.817601],
                                        desc="SF66", long_desc="N-SF66 (SCHOTT)")
"""N-SF66 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-SF&page=N-SF66"""


SK14: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.936155374, 0.00461716525, 0.594052018,
                                                            0.016885927, 1.04374583, 103.736265],
                                        desc="SK14", long_desc="N-SK14 (SCHOTT)")
"""N-SK14 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-SK&page=N-SK14"""


# https://refractiveindex.info/?shelf=3d&book=glass&page=soda-lime-clear
Soda_Lime: RefractionIndex = RefractionIndex("Function",
                                             func=lambda x: 1.5130-0.003169 *
                                             (x*1e-3)**2+0.003962*(x*1e-3)**-2,
                                             desc="Soda Lime",
                                             long_desc="Clear soda lime silica window glass")
"""Clear soda lime silica window glass"""


SSK8: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.44857867, 0.00869310149, 0.117965926,
                                                            0.0421566593, 1.06937528, 111.300666],
                                        desc="SSK8", long_desc="N-SSK8 (SCHOTT)")
"""N-SSK8 (SCHOTT), see https://refractiveindex.info/?shelf=glass&book=SCHOTT-SSK&page=N-SSK8"""


glasses: list = [BAF10, BAK1, BASF64, BK7, F2, FK51A, Fused_Silica, K5, LAF2, LAK8, LAK22,
                 LASF9, LASF44, LF5, SF5, SF6, SF10, SF11, SF66, SK14, Soda_Lime, SSK8]
"""glass refraction index presets"""


# Plastics
#######################################################################################################################

CR39: RefractionIndex = RefractionIndex("Conrady", coeff=[1.471862713E+000, 1.520790642E-002, 3.555509148E-005],
                                        desc="CR39", long_desc="CR-39, PADC, Poly(allyl diglycol carbonate)")
"""CR-39, PADC, Poly(allyl diglycol carbonate), see Conrady equation coefficients from ZEMAX model for CR39, 
thread https://community.zemax.com/got-a-question-7/cr-39-eyeglass-lenses-677
attached file ophthalmic.zip, line 50 in ophthalmic.agf"""


COC: RefractionIndex = RefractionIndex("Sellmeier", coeff=[2.045-1., 0, 0.266, 0.206**2], desc="COC",
                                       long_desc="Topas COC 5013 at 25°C")
"""Topas COC 5013 at 25°C, 
see G. Khanarian, Hoechst Celanese - "Optical properties of cyclic olefin copolymers", Table 1"""


Finalite: RefractionIndex = RefractionIndex("Abbe", n=1.600, V=42, desc="Finalite", long_desc="Sola Finalite")
"""
Sola Finalite,
from https://eyewiki.aao.org/Lens_Material_Properties
"""


MR7: RefractionIndex = RefractionIndex("Abbe", n=1.660, V=32, desc="MR-7", long_desc="MR-7")
"""
MR-7,
from https://eyewiki.aao.org/Lens_Material_Properties
"""

Ormex: RefractionIndex = RefractionIndex("Abbe", n=1.558, V=32, desc="Ormex", long_desc="Essilor Ormex")
"""
Essilor Ormex,
from https://eyewiki.aao.org/Lens_Material_Properties
"""


PC: RefractionIndex = RefractionIndex(
    "Sellmeier", coeff=[1.4182, 0.021304], desc="PC", long_desc="Polycarbonate")
"""Polycarbonate, see # https://refractiveindex.info/?shelf=organic&book=polycarbonate&page=Sultanova"""


PDSM: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.0057, 0.013217], desc="PDSM",
                                        long_desc="Polydimethylsiloxane")
"""Polydimethylsiloxane, 
see https://refractiveindex.info/?shelf=organic&book=polydimethylsiloxane&page=Schneider-RTV615"""


PET: RefractionIndex = RefractionIndex("Data", wls=np.concatenate(([380], 400+10*np.arange(39))),
                                       vals=[1.61891, 1.61027, 1.60595, 1.60212, 1.59847, 1.59528, 1.59247, 1.58988,
                                             1.58716, 1.58496, 1.58304, 1.58111, 1.57927, 1.57769, 1.57630, 1.57470,
                                             1.57333, 1.57194, 1.57086, 1.56993, 1.56904, 1.56811, 1.56696, 1.56627,
                                             1.56527, 1.56478, 1.56368, 1.56317, 1.56225, 1.56199, 1.56131, 1.56052,
                                             1.56013, 1.55933, 1.55868, 1.55854, 1.55817, 1.55795, 1.55723, 1.55583],
                                       desc="PET", long_desc="Polyethylene terephthalate")
"""Polyethylene terephthalate, 
see https://refractiveindex.info/?shelf=organic&book=polyethylene_terephthalate&page=Zhang
linearly extrapolated from 400nm to 380nm"""


PMMA: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.99654, 0.00787, 0.18964, 0.02191,
                                                            0.00411, 3.85727],
                                        desc="PMMA", long_desc="Poly(methyl methacrylate)")
"""Poly(methyl methacrylate), 
see https://refractiveindex.info/?shelf=organic&book=poly%28methyl_methacrylate%29&page=Szczurowski"""


PS: RefractionIndex = RefractionIndex("Sellmeier", coeff=[1.4435, 0.020216],
                                      desc="PS", long_desc="Polystyren")
"""Polystyren, see https://refractiveindex.info/?shelf=3d&book=plastics&page=ps"""


Spectralite: RefractionIndex = RefractionIndex("Abbe", n=1.537, V=47, desc="Spectralite", long_desc="Sola Spectralite")
"""
Sola Spectralite,
from https://eyewiki.aao.org/Lens_Material_Properties
"""


plastics: list = [CR39, COC, Finalite, MR7, Ormex, PC, PDSM, PET, PMMA, PS, Spectralite]
"""plastics refraction index presets"""


# Misc
#######################################################################################################################

Air: RefractionIndex = RefractionIndex("Constant", n=1.00027784, desc="Air", long_desc="Air at 550nm, 15°C, 1013.25hPa")
"""Air at 550nm, 15°C, 1013.25hPa, neglecting dispersion, 
see https://refractiveindex.info/?shelf=other&book=air&page=Ciddor"""


BaF2: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.643356, 0.057789**2, 0.506762, 0.10968**2,
                                                            3.8261, 46.3864**2],
                                        desc="BaF2", long_desc="BaF2 (Barium fluoride)")
"""BaF2 (Barium fluoride), see https://refractiveindex.info/?shelf=main&book=BaF2&page=Malitson"""


CaF2: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.5675888, 0.050263605**2, 0.4710914,
                                                            0.1003909**2, 3.8484723, 34.649040**2],
                                        desc="CaF2", long_desc="CaF2 (Calcium fluoride)")
"""CaF2 (Calcium fluoride), see https://refractiveindex.info/?shelf=main&book=CaF2&page=Malitson"""


Diamond: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.3306, 0.1750**2, 4.3356, 0.1060**2],
                                           desc="Diamond", long_desc="Diamond")
"""Diamond, see https://refractiveindex.info/?shelf=main&book=C&page=Peter"""


Ethanol: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.0165, 9.08, 0.8268, 0.01039],
                                           desc="Ethanol", long_desc="C2H5OH (Ethanol)")
"""C2H5OH (Ethanol), see https://refractiveindex.info/?shelf=organic&book=ethanol&page=Sani-formula"""


Ice: RefractionIndex = RefractionIndex("Data", wls=380+10*np.arange(41),
                                       vals=[1.32145, 1.3203, 1.3194, 1.3185, 1.3177, 1.3170, 1.3163, 1.3157,
                                             1.3151, 1.3145, 1.3140, 1.3135, 1.3130, 1.3126, 1.3121, 1.3117,
                                             1.3114, 1.3110, 1.3106, 1.3103, 1.3100, 1.3097, 1.3094, 1.3091,
                                             1.3088, 1.3085, 1.3083, 1.3080, 1.3078, 1.3076, 1.3073, 1.3071,
                                             1.3069, 1.3067, 1.3065, 1.3062, 1.3060, 1.3059, 1.3057, 1.3055,
                                             1.3053], desc="Ice", long_desc="Water Ice at -7°C")
"""Water Ice at -7°C, 
see https://refractiveindex.info/?shelf=3d&book=crystals&page=ice (from table in full database record)
interpolated so it starts at 380nm
"""


MgF2: RefractionIndex = RefractionIndex("Sellmeier", coeff=[0.48755108, 0.04338408**2, 0.39875031,
                                                            0.09461442**2, 2.3120353, 23.793604**2],
                                        desc="MgF2", long_desc="MgF2 (Magnesium fluoride)")
"""MgF2 (Magnesium fluoride), see https://refractiveindex.info/?shelf=main&book=MgF2&page=Dodge-o"""


Vacuum: RefractionIndex = RefractionIndex(
    "Constant", n=1.0, desc="Vacuum", long_desc="Vacuum")
"""Vacuum"""


Water: RefractionIndex = RefractionIndex("Sellmeier", coeff=[5.684027565E-1, 5.101829712E-3, 1.726177391E-1,
                                                             1.821153936E-2, 2.086189578E-2, 2.620722293E-2,
                                                             1.130748688E-1, 1.069792721E1],
                                         desc="Water", long_desc="Water at 20.0°C")
"""Water at 20.0°C, see https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon-20.0C """


misc: list = [Air, BaF2, CaF2, Diamond, Ethanol, Ice, MgF2, Vacuum, Water]
"""misc refraction index presets"""


# list of all n presets
#######################################################################################################################

all_presets: list = [*glasses, *plastics, *misc]
"""all refraction index presets in one list"""
