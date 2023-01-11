# Test source SN computation
from pyetc.wst import WST
from pyetc.etc import peakwave_asymgauss
from numpy.testing import assert_almost_equal

wst = WST(log='DEBUG')
ifs = wst.ifs['red']
wave = 8000 # central wavelength A
ima_fwhm = 1 # image moffat fwhm arcsec
ima_beta = 2.0 # image moffat beta
spec_sigma = 4.0 # spectral shape sigma in A
spec_skew = 7.0 # spectral shape skewness
ifs = wst.ifs['red']

ima,spec = wst.get_source(ifs, wave, ima_fwhm, ima_beta, spec_sigma, 
                          spec_skew, spec_kfwhm=10, ima_kfwhm=10)
lmax = spec.wave.coord(spec.data.argmax())
assert abs(lmax-wave)<spec.get_step()
assert_almost_equal(ima.data.sum(), 1)
assert_almost_equal(spec.data.sum(), 1, decimal=4)

obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    kfwhm_spec = 1.8, # spectral half window adaptive size 
    kfwhm_ima = 1.8, # spatial half window adaptive circular aperture (only IFS)
    airmass = 1,
)
wst.set_obs(obs)

ifs = wst.ifs['red']
moon = 'darksky'
flux = 5.e-18
ima,spec = wst.get_source(ifs, wave, ima_fwhm, ima_beta, spec_sigma, spec_skew)
res = wst.snr_from_source(ifs, flux, ima, spec, moon)
aper = res['aper']
assert aper['frac_flux'] > 0
assert aper['frac_flux'] == aper['frac_ima'] * aper['frac_spec']
assert aper['nvoxels'] > 0
assert aper['nvoxels'] == aper['nspaxels'] * aper['nspectels']
assert aper['snr'] > 0
snr = aper['snr']

res = wst.flux_from_source(ifs, snr, ima, spec, moon)
assert_almost_equal(res['aper']['snr'],snr,decimal=3)
assert_almost_equal(res['flux'],flux,decimal=3)

mos = wst.moslr['red']
ima,spec = wst.get_source(mos, wave, ima_fwhm, ima_beta, spec_sigma, spec_skew)
obs = dict(
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    kfwhm_spec = 1.8, # spectral half window adaptive size 
    airmass = 1.2,
)
wst.set_obs(obs)
res = wst.snr_from_source(mos, flux, ima, spec, moon)
aper = res['aper']
assert aper['frac_flux'] > 0
assert aper['frac_flux'] == aper['frac_ima'] * aper['frac_spec']
assert aper['nvoxels'] > 0
assert aper['nvoxels'] == aper['nspaxels'] * aper['nspectels']
assert aper['snr'] > 0
snr = aper['snr']

res = wst.flux_from_source(mos, snr, ima, spec, moon)
assert_almost_equal(res['aper']['snr'],snr,decimal=3)
assert_almost_equal(res['flux'],flux,decimal=3)


