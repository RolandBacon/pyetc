# Test source resolved and line
import numpy as np
from pyetc.wst import WST
from pyetc.etc import peakwave_asymgauss, compute_sky
from numpy.testing import assert_almost_equal, assert_allclose

wst = WST(log='DEBUG')
ifs = wst.ifs['red']

# --------------
wave = 7000
dspec = dict(type='line', lbda=wave, sigma=5.0/2.355, skew=0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=0.7, beta=8.0)
ima = wst.get_ima(ifs, dima)
res = ima.gauss_fit(unit_center=None, unit_fwhm=None)
fwhm = res.fwhm[0]*ifs['spaxel_size']/ima.oversamp
assert_allclose(fwhm, 0.70, rtol=0.1)
res = spec.gauss_fit(7000-200,7000+200)
assert_allclose(res.fwhm, 5.0, rtol=0.1)
lmax = spec.wave.coord(spec.data.argmax())
assert abs(lmax-wave)<spec.get_step()
assert_almost_equal(ima.data.sum(), 1, decimal=3)
assert_almost_equal(spec.data.sum(), 1, decimal=3)

# ------------------------------
wave = 8000 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=1.0, beta=2.0)
ima = wst.get_ima(ifs, dima, uneven=0)

obs = dict(
    airmass = 1,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'fixed',
    spec_range_hsize_spectels = 3,
    ima_type = 'resolved', 
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 2, # spatial half window adaptive circular aperture (only IFS)   
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
res0 = wst.snr_from_source(ifs, flux, ima, spec, moon)
kfwhm = wst.optimum_circular_aperture(ifs, flux, ima, spec, moon)
assert kfwhm > 0
res1 = wst.snr_from_source(ifs, flux, ima, spec, moon)
assert res1['aper']['snr'] > res0['aper']['snr']

#--------------------------------

wave = 8000 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=1.0, beta=2.0)
ima = wst.get_ima(ifs, dima, uneven=0)

obs = dict(
    airmass = 1,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'adaptative',
    spec_range_kfwhm = 3,
    ima_type = 'resolved', 
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 7.4, # spatial half window adaptive circular aperture (only IFS)   
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
res0 = wst.snr_from_source(ifs, flux, ima, spec, moon)
kfwhm = wst.optimum_spectral_range(ifs, flux, ima, spec, moon)
assert kfwhm > 0
res1 = wst.snr_from_source(ifs, flux, ima, spec, moon)
assert res1['aper']['snr'] > res0['aper']['snr']


#--------------------------------

wave = 8000 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=1.0, beta=2.0)
ima = wst.get_ima(ifs, dima, uneven=0)

obs = dict(
    airmass = 1,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'adaptative',
    spec_range_kfwhm = 0.94,
    ima_type = 'resolved', 
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 7.4, # spatial half window adaptive circular aperture (only IFS)   
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
res0 = wst.snr_from_source(ifs, flux, ima, spec, moon)
snr0 = res0['aper']['snr']
res1 = wst.flux_from_source(ifs, snr0, ima, spec, moon)
aper = res1['aper']
assert_allclose(flux, aper['flux'], rtol=0.01)
assert aper['frac_flux'] > 0
assert aper['frac_flux'] == aper['frac_ima'] * aper['frac_spec']
assert aper['nb_voxels'] > 0
assert aper['nb_voxels'] == aper['nb_spaxels'] * aper['nb_spectels']
assert aper['snr'] > 0
assert_almost_equal(aper['snr'],snr0,decimal=3)

#----------------

mos = wst.moslr['red']
wave = 8000 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=0.7, beta=2.5)
ima = wst.get_ima(ifs, dima, uneven=0)
obs = dict(
    airmass = 1,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'adaptative',
    spec_range_kfwhm = 0.94,
    ima_type = 'resolved', 
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
res = wst.snr_from_source(mos, flux, ima, spec, moon)
aper = res['aper']
assert res['spec']['snr'].shape[0] > 0
assert aper['frac_flux'] > 0
assert aper['frac_flux'] == aper['frac_ima'] * aper['frac_spec']
assert aper['nb_voxels'] > 0
assert aper['nb_voxels'] == aper['nb_spaxels'] * aper['nb_spectels']
assert aper['snr'] > 0

#-------------------------
ifs = wst.ifs['blue']
wave = 4100 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=0.7, beta=2.5)
ima = wst.get_ima(ifs, dima, uneven=0)
obs = dict(
    airmass = 2.0,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'adaptative',
    spec_range_kfwhm = 0.94,
    ima_type = 'resolved', 
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 5, # spatial half window adaptive circular aperture (only IFS)       
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
wst.obs['airmass'] = 1.0
res1 = wst.snr_from_source(ifs, flux, ima, spec, moon)
wst.obs['airmass'] = 1.2
res2 = wst.snr_from_source(ifs, flux, ima, spec, moon)
assert res1['aper']['snr'] > res2['aper']['snr']
assert res1['aper']['nph_source'] > res2['aper']['nph_source']







