# Test main
from pyetc.wst import WST
from pyetc.etc import asymgauss, vdisp2sigma, sigma2vdisp, moffat
from mpdaf.obj import Spectrum, WaveCoord
import numpy as np


print('start')
etc = WST(log='INFO')

ifs = etc.ifs['blue']
obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    airmass = 1,
    seeing = 0.7,
    beta = 2.8,
    nradius_spaxels=1,
)
etc.set_obs(obs)
flux = 1e-18 # input Surface brightness in erg/s/cm2
vdisp = 170 
skew = 3.0 
z = 3
l0 = 1216*(1+z)
sigma = vdisp2sigma(vdisp, l0)
fwhm,beta = 0.8,2.8
ima,sp = etc.get_source(l0, fwhm, beta, sigma, skew, 'blue', 
                   ima_kfwhm=3, spec_kfwhm=2)

snr = 3.0
res = etc.flux_from_source(snr, ima, sp, 'darksky', 'blue')


print('end')
#etc.info()
#obs = dict(
    #ndit = 2, # number of exposures
    #dit = 1800, # exposure time in sec of one exposure
    #nb_spaxels = 4 * 4, # number of spaxels
    #nb_spectels = 3, # number of spectral pixels
    #airmass = 1,
#)
#etc.set_obs(obs)
#sbflux = 1e-18 # input Surface brightness in erg/s/cm2/arcsec2/A
#res = etc.snr_from_sb(sbflux, 'darksky', 'blue')
#snr = 5
#res = etc.sb_from_snr(snr, 'darksky', 'blue')
#mag = 25.0
#res = etc.snr_from_esmag(mag, 'darksky', 'blue')
#snr = 2
#res = etc.esmag_from_snr(snr, 'darksky', 'blue')
#obs = dict(
    #ndit = 2, # number of exposures
    #dit = 1800, # exposure time in sec of one exposure
    #nb_spectels = 3, # number of spectral pixels
    #airmass = 1,
    #seeing = 0.7,
    #nradius_spaxels = 2,
#)
#etc.set_obs(obs)
#flux = 1e-18
#res = etc.snr_from_psflux(flux, 'darksky', 'blue')
#res = etc.snr_from_psmag(flux, 'darksky', 'blue')
#snr = 3
#res = etc.psflux_from_snr(snr, 'darksky', 'blue')
#res = etc.psmag_from_snr(snr, 'darksky', 'blue')

obs = dict(
    ndit = 20, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    nb_spectels = 3, # number of spectral pixels
    airmass = 1,
    seeing = 0.7, # seeing in arcsec at zenith
    nradius_spaxels = 2, # half size of aperture in spaxels (2*n+1)
)
etc.set_obs(obs)
flux = 1e-19
res = etc.snr_from_psflux(flux, 'darksky', 'blue')
print('end')
