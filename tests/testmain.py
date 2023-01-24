# Test main

from pyetc.vlt import VLT
from pyetc.wst import WST
from mpdaf.obj import Spectrum, WaveCoord, mag2flux, flux2mag
import numpy as np
from matplotlib import pyplot as plt


print('start')
wst = WST(log='DEBUG')
#wst.info()

mos = wst.moslr['blue']
wave,dw  = 5500,500
dspec = dict(type='template', name='ref/sun', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(mos, dspec)
mag = 23
flux = mag2flux(mag, wave)
obs = dict(
    moon = 'brightsky',
    seeing = 0.7,
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
)
wst.set_obs(obs)

res = wst.snr_from_source(mos, flux, None, spec)


print('end')
