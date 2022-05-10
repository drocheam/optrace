from optrace.tracer.RefractionIndex import *

# air has very low dispersion, using a constant value speeds things up
# while not being too incorrect at the same time
# Air at 550nm, 15 °C pressure: 101325 Pa, 
# see https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
preset_n_Air = RefractionIndex("Constant", n=1.00027784, desc="Air")

# https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT 
preset_n_BK7 = RefractionIndex("Sellmeier", coeff=[1.03961212, 0.00600069867, 
                                       0.231792344, 0.0200179144, 1.01046945, 103.560653], desc="BK7")

# https://refractiveindex.info/?shelf=glass&book=BAF10&page=SCHOTT
preset_n_BAF10 = RefractionIndex("Sellmeier", coeff=[1.5851495, 0.00926681282, 
                                         0.143559385, 0.0424489805, 1.08521269, 105.613573], desc="BAF10")

# https://refractiveindex.info/?shelf=glass&book=BAK1&page=SCHOTT
preset_n_BAK1 = RefractionIndex("Sellmeier", coeff=[1.12365662, 0.00644742752, 0.309276848, 
                                        0.0222284402, 0.881511957, 107.297751], desc="BAK1")

# Conrady equation coefficients from ZEMAX model for CR39, 
# see thread https://community.zemax.com/got-a-question-7/cr-39-eyeglass-lenses-677
# attached file ophthalmic.zip, line 50 in ophthalmic.agf
preset_n_CR39 = RefractionIndex("Conrady", coeff=[1.471862713E+000, 1.520790642E-002,
                                        3.555509148E-005], desc="CR39")

# https://refractiveindex.info/?shelf=glass&book=FK51A&page=SCHOTT
preset_n_FK51A = RefractionIndex("Sellmeier", coeff=[0.971247817, 0.00472301995, 0.216901417, 
                                         0.0153575612, 0.904651666, 168.68133], desc="FK51A")
            
# https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
preset_n_Fused_Silica = RefractionIndex("Sellmeier", coeff=[0.6961663, 0.0684043**2, 0.4079426, 
                                                0.1162414**2, 0.8974794, 9.896161**2], desc="Fused_Silica")

# https://refractiveindex.info/?shelf=glass&book=SCHOTT-K&page=N-K5
preset_n_K5 = RefractionIndex("Sellmeier", coeff=[1.08511833, 0.00661099503, 0.199562005, 
                                      0.024110866, 0.930511663, 111.982777], desc="K5") 

# https://refractiveindex.info/?shelf=glass&book=LASF9&page=SCHOTT
preset_n_LASF9 = RefractionIndex("Sellmeier", coeff=[2.00029547, 0.0121426017, 0.298926886, 
                                         0.0538736236, 1.80691843, 156.530829], desc="LASF9") 

# https://refractiveindex.info/?shelf=organic&book=polycarbonate&page=Sultanova
preset_n_PC = RefractionIndex("Sellmeier", coeff=[1.4182, 0.021304], desc="PC") 

# https://refractiveindex.info/?shelf=organic&book=poly%28methyl_methacrylate%29&page=Szczurowski
preset_n_PMMA = RefractionIndex("Sellmeier", coeff=[0.99654, 0.00787, 0.18964, 0.02191, 
                                        0.00411, 3.85727], desc="PMMA") 

# https://refractiveindex.info/?shelf=glass&book=SF5&page=SCHOTT
preset_n_SF5 = RefractionIndex("Sellmeier", coeff=[1.52481889, 0.011254756, 0.187085527, 
                                       0.0588995392, 1.42729015, 129.141675], desc="SF5") 
                        
# https://refractiveindex.info/?shelf=glass&book=SF10&page=SCHOTT
preset_n_SF10 = RefractionIndex("Sellmeier", coeff=[1.62153902, 0.0122241457, 0.256287842, 
                                        0.0595736775, 1.64447552, 147.468793],desc="SF10") 

# https://refractiveindex.info/?shelf=glass&book=SF11&page=SCHOTT
preset_n_SF11 = RefractionIndex("Sellmeier", coeff=[1.73759695, 0.013188707, 0.313747346, 
                                        0.0623068142, 1.89878101, 155.23629], desc="SF11") 

preset_n_Vacuum = RefractionIndex("Constant", n=1.0, desc="Vacuum")

# https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon-20.0C
preset_n_Water = RefractionIndex("Sellmeier", coeff=[5.684027565E-1, 5.101829712E-3, 
                                         1.726177391E-1, 1.821153936E-2, 2.086189578E-2, 2.620722293E-2, 
                                         1.130748688E-1, 1.069792721E1], desc="Water")

presets_n = [preset_n_Air, preset_n_BK7, preset_n_BAF10, preset_n_BAK1, preset_n_CR39,
             preset_n_FK51A, preset_n_Fused_Silica, preset_n_K5, preset_n_LASF9, preset_n_PC,
             preset_n_PMMA, preset_n_SF5, preset_n_SF10, preset_n_SF11, preset_n_Vacuum, preset_n_Water]

