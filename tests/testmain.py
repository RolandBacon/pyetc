# Test main
from pyetc.wst import WST
from pyetc.etc import asymgauss, vdisp2sigma, sigma2vdisp, moffat
from mpdaf.obj import Spectrum, WaveCoord
import numpy as np


print('start')
wst = WST(log='INFO')
wst.info()



obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    nb_spectels = 5, # number of spectral pixels
    airmass = 1,
    frac_flux = 0.8, # for ifs only
    seeing = 0.7, # seeing in arcsec at zenith    
)
wst.set_obs(obs)
flux = 5.e-18
moon = 'darksky'
ifs = wst.ifs['red']
res = wst.snr_from_psflux(ifs, flux, moon)
k = res['snr'].wave.pixel(8000, nearest=True)
print('IFS S/N:', res['snr'].data[k])
print('nph_source:', res['nph_source'].data[k])


wave = 8000 # central wavelength A
ima_fwhm = 1 # image moffat fwhm arcsec
ima_beta = 2.0 # image moffat beta
spec_sigma = 4.0 # spectral shape sigma in A
spec_skew = 7 # spectral shape skewness

mos = wst.moslr['red']
ima,spec = wst.get_source(mos, wave, ima_fwhm, ima_beta, spec_sigma, 
                          spec_skew, spec_kfwhm=5, ima_kfwhm=5)

#tspec = wst.truncate_spec(mos, spec, 1.8)
tima = wst.fixed_circular_aperture(mos, ima, 0.5)
from pyetc.etc import _adaptative_circular_aperture
wave = 8000 # central wavelength A
ima_fwhm = 1 # image moffat fwhm arcsec
ima_beta = 2.0 # image moffat beta
spec_sigma = 4.0 # spectral shape sigma in A
spec_skew = 7 # spectral shape skewness
ifs = wst.ifs['red']
ima,spec = wst.get_source(ifs, wave, ima_fwhm, ima_beta, spec_sigma, 
                          spec_skew, spec_kfwhm=5)
tima = _adaptative_circular_aperture(ima, 3)


obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    nb_spectels = 3, # number of spectral pixels
    airmass = 1,
    seeing = 0.7, # seeing in arcsec at zenith
    nradius_spaxels = 2, # half size of aperture in spaxels (2*n+1)
)
wst.set_obs(obs)
res = wst.snr_from_psflux(wst.moslr['blue'], 1.e-18, 'darksky')
ifs = wst.ifs['blue']
sbflux = 1e-18 # input emission line surface brightness in erg/s/cm2/arcsec2      

res1 = wst.snr_from_sb(ifs, sbflux, 'darksky')
k = 2000
snr0 = res1['snr'].data[k]
res2 = wst.sb_from_snr(ifs, snr0, 'darksky')
sbflux0 = res2['sbflux'].data[k]

obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    airmass = 1,
    seeing = 0.7,
    beta = 2.8,
    nradius_spaxels=1,
)
etc.set_obs(obs)
sbflux = 1e-18 # input Surface brightness in erg/s/cm2/arcsec2
res = etc.snr_from_sb(etc.ifs['blue'], sbflux, 'darksky')

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
