# Test main

from pyetc.etc import vdisp2sigma
from pyetc.wst import WST
from mpdaf.obj import Spectrum, WaveCoord, mag2flux, flux2mag
import numpy as np
from matplotlib import pyplot as plt



print('start')
wst = WST(log='DEBUG')

mosr = wst.moslr['red']

obs = dict(
    moon = 'darksky',
    airmass = 1,
    seeing = 0.7,
    ndit = 2,
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
)
wst.set_obs(obs)

snr = 3 #by A
wave = 8000
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = wst.get_spec(mosr, dspec)
res = wst.flux_from_source(mosr, snr, None, spec)
print('end')
