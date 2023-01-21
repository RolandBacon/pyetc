# Comparison test with other ETC
from pyetc.vlt import VLT
from numpy.testing import assert_almost_equal, assert_allclose
from mpdaf.obj import flux2mag, mag2flux
import astropy.units as u

etc = VLT(log='INFO')

# Comparison with MUSE
# https://eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr
# target emission line wave 700 nm Flux 0.05e-16 erg/s/cm2 FWHM 0.5 nm point source
# Nb of co-added spatial pixel 5x5 spectral 5
# sky airmass 1.0 PWC 10 nm PWV Proba 95% Seeing 0.79 arcsec @500 nm, zenith
# Sun-Moon 90° FLI 0 Moon-target 45° Moon alt 45°
# FWHM IQ @ 700nm 0.70 arcsec
# Inst WFM NOAO RON 3e-/pixel/DIT Dark 3 e-/pixel/hour Disp 0.125 nm/pixel scale 0.2 arcsec
# DIT 1800s NDIT 2
# S/N per spectral bin 7.69
# Peak pixel value obj+sky 41.1 e-/DIT S/N ref area 0.36 arcsec**2 Nspec in spectral bin 
# Target signal in S/N ref area per spec bin: 429 e-
# Sky signal in S/N ref area: 6226 e-
# System transmission without atm: 32.29%
# -----------------------------------------
ifs = etc.ifs['red']
wave = 7000
trans = ifs['instrans']
k = trans.wave.pixel(wave, nearest=True)
tval = trans.data[k]
assert_almost_equal(tval, 0.323, decimal=2) # 0.33
obs = dict(
    airmass = 1,  
    ndit = 2, 
    dit = 1800, 
    spec_type = 'line',
    spec_range_type = 'fixed',
    spec_range_hsize_spectels = 2,
    ima_type = 'resolved', 
    ima_aperture_type = 'square_fixed',
    ima_aperture_hsize_spaxels = 2, # in spaxels   
)
etc.set_obs(obs)
dspec = dict(type='line', lbda=wave, sigma=2.5/2.355, skew=0)
spec = etc.get_spec(ifs, dspec, lsfconv=False)
dima = dict(type='moffat', fwhm=0.7, beta=8.0)
ima = etc.get_ima(ifs, dima)
moon = 'darksky'
flux = 5.e-18
res = etc.snr_from_source(ifs, flux, ima, spec, moon)
aper = res['aper']
assert aper['nb_spaxels'] == 25
assert aper['nb_spectels'] == 5
assert_allclose(aper['nph_sky'], 6226, rtol=0.1) # 6539
assert_allclose(aper['nph_source'], 753, rtol=0.1) #702
assert_allclose(aper['snr'], 7.69, rtol=0.1) #7.07

# Comparison with MUSE
# https://eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr
# target continuum waveref 800 nm AB R 23 arcsec-2 
# Nb of co-added spatial pixel 5x5 spectral 1
# sky airmass 1.2 
# Sun-Moon 90° FLI 0 Moon-target 45° Moon alt 45°
# Inst WFM NOAO RON 3e-/pixel/DIT Dark 3 e-/pixel/hour Disp 0.125 nm/pixel scale 0.2 arcsec
# DIT 3600s NDIT 1
# S/N 6.41 per spectral bin 
# Peak pixel value obj+sky 103.4 e-/DIT S/N ref area 1 arcsec**2 Nspec 1 in spectral bin 
# Target signal in S/N ref area per spec bin: 340 e-
# Sky signal in S/N ref area: 2170 e-
# System transmission without atm: 24.2%
# -----------------------------------------
ifs = etc.ifs['red']
wave = 8000
trans = ifs['instrans']
k = trans.wave.pixel(wave, nearest=True)
tval = trans.data[k]
assert_almost_equal(tval, 0.242, decimal=2) # OK
obs = dict(
    airmass = 1.2,  
    ndit = 1, 
    dit = 3600, 
    spec_type = 'cont',
    ima_type = 'sb', 
    ima_area = 1, # arcsec**2  
)
etc.set_obs(obs)
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = etc.get_spec(ifs, dspec)
moon = 'darksky'
mag = 23
flux = mag2flux(mag, 6400)
res = etc.snr_from_source(ifs, flux, None, spec, moon)
sp = res['spec']
assert sp['nb_spaxels'] == 25
assert sp['nb_spectels'] == 1
k = sp['snr'].wave.pixel(wave, nearest=True)
assert_allclose(sp['nph_sky'].data[k], 2071, rtol=0.1) # 2078 
assert_allclose(sp['nph_source'].data[k], 340, rtol=0.1) # 346
assert_allclose(sp['snr'].data[k], 6.41, rtol=0.1) # 6.63

# Comparison with BlueMUSE
# spaxel=0.2 lstep=0.575 lsf=lstep*2.0
# dit=3600s ndit=10 lref=4500
# airmass=1 seeing=0.8 moon='d' nspatial=5 nspectral=1 
# fwhmline 5.2875 fluxline 1e-18
# iq 0.735 arcsec frac 0.7661 transm inc sky 0.3596 @4500 A
# use moffat beta=2.5 for psf shape
# skyelectrons 25534 signalelectrons 382
# frac_ima 0.561 frac_spec 0.478
# TARGET SNR per 2.88 Angstroms spectral bin at reference wavelength: 1.86
# -----------------------------------------
ifs = etc.ifs['blue']
assert ifs['dlbda'] == 0.575
wave = 4500
trans = ifs['instrans']
moon = 'darksky'
k = trans.wave.pixel(wave, nearest=True)
emi_sky,abs_sky = etc.get_sky(ifs, moon)
tval = trans.data[k] * abs_sky.data[k]
assert_allclose(tval, 0.3596, rtol=0.05) # 0.33
obs = dict(
    airmass = 1,
    seeing = 0.735,
    beta = 2.5,
    ndit = 10, 
    dit = 3600, 
    spec_type = 'line',
    spec_range_type = 'fixed',
    spec_range_hsize_spectels = 2,
    ima_type = 'resolved', 
    ima_aperture_type = 'square_fixed',
    ima_aperture_hsize_spaxels = 2,        
)
etc.set_obs(obs)
dspec = dict(type='line', lbda=wave, sigma=5.2875/2.355, skew=0)
spec = etc.get_spec(ifs, dspec, lsfconv=False)
dima = dict(type='moffat', fwhm=0.735, beta=2.5)
ima = etc.get_ima(ifs, dima)
moon = 'darksky'
flux = 1.e-18 
res = etc.snr_from_source(ifs, flux, ima, spec, moon)
aper = res['aper']
assert aper['nb_spaxels'] == 25
assert aper['nb_spectels'] == 5
assert_allclose(aper['nph_sky'], 25534, rtol=0.1) # 25432
assert_allclose(aper['frac_ima'], 0.561, rtol=0.1) # 0.578
assert_allclose(aper['frac_spec'], 0.478, rtol=0.1) # 0.466
assert_allclose(aper['nph_source'], 382, rtol=0.1) # 383
assert_allclose(aper['snr'], 1.86, rtol=0.1) # 1.89


# Comparison with GIRAFFE MOS Medusa
# Source flux distribution uniform AB=17 Flux=5.843 10**5 W.m-2.microns-1
# Sky airmass 1 Moon FLI 0
# Image quality 0.8 arcsec
# Object-fiber displacement 0, Detector bin 1
# dit 1800 s nit 1
# L-inst LR01 362.349 - 408.089  
# Ref wavelngth 385.7 nm
# Dispersion 0.012 nm/pixel  Plate scale 0.30 arcsec/pixel
# FWHM fiber spatial profile 4 pixels
# at ref wavelength efficiency (no extinction) 1.83% with extinction 1.285% Fiber injection loss 20.832%
# at ref wavelength total object signal 126.2 e- Sky background 2.2 e- Max intensity 29.7
# Detect ron 4 e-/pixel dark 0.5 e-/pixel/hour
# Fiber diameter 6 pixels
# SN at ref 8.423
mos = etc.giraffe['blue']
obs = dict(
    airmass = 1.0,
    ndit = 1, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'resolved',
)
etc.set_obs(obs)

ima_fwhm = 0.8 # image moffat fwhm arcsec
ima_beta = 4.0 # image moffat beta
spec_sigma = 10.0/2.355 # spectral shape sigma in A
spec_skew = 0 # spectral shape skewness
flux = 5.843e5 * 1.e-20 * (1000/1.e4) # en erg/s/cm2/A
mag = flux2mag(flux, 0, 5450)[0]
assert_allclose(mag, 17.0, rtol=0.01)
wave = 3857
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = etc.get_spec(mos, dspec, oversamp=10)
dima = dict(type='moffat', fwhm=0.8, beta=2.5, kfwhm=5)
ima = etc.get_ima(mos, dima, uneven=0)
moon = 'greysky'
res = etc.snr_from_source(mos, flux, ima, spec, moon)
aper = res['spec']
assert aper['nb_spectels'] == 1
#assert aper['nb_spaxels'] == 29 # pi*6**2/4 = 28.7 get 32
k = aper['snr'].wave.pixel(wave, nearest=True)
emi_sky,abs_sky = etc.get_sky(mos, moon)
assert_allclose(mos['instrans'].data[k], 0.0183, 0.1) # 0.0187
assert_allclose(mos['instrans'].data[k]*abs_sky.data[k], 0.0129, 0.1) # 0.0124
assert_allclose(1-aper['frac_flux'], 0.21, 0.1) # 0.21
#assert_allclose(aper['snr'].data[k], 8.423, 0.1) # 6.26
#assert_allclose(aper['nph_source'].data[k], 126.2, 0.1) # 168
#assert_allclose(aper['nph_sky'].data[k]/aper['nb_spaxels'], 2.2, 0.1) # 0.94

