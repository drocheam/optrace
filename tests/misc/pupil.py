#!/usr/bin/env python3

import numpy as np
import optrace as ot


IL = ot.IdealLens(r=2, D=10, pos=[0, 0, 0])
IL2 = ot.IdealLens(r=2, D=1/0.075, pos=[0, 0, 50])

tma = ot.TMA([IL, IL2])

zp = 25
print(tma.pupil_position(zp))
print(tma.pupil_magnification(zp))
