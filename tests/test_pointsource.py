# Test point source SN computation
from pyetc.wst import WST
from pyetc.etc import get_seeing_fwhm
from numpy.testing import assert_almost_equal, assert_allclose
from mpdaf.obj import flux2mag, mag2flux

wave = 5500
fwhm = get_seeing_fwhm(0.8, 1.0, wave, 8.2, 0)
assert_allclose(fwhm, 0.655, rtol=0.2)



wst = WST(log='DEBUG')

ifs = wst.ifs['red']
mos = wst.moslr['red']

waves = ifs['wave'].coord()
fwhm = get_seeing_fwhm(0.8, 1.2, waves, wst.tel['diameter'], ifs['iq_fwhm'])
assert fwhm[0] > fwhm[-1]

obs = dict(
    airmass = 1.0,
    seeing = 0.7,
    )
wst.set_obs(obs)
iq = wst.get_image_quality(ifs)
assert iq.data.min() < 0.7
assert iq.data[0] > iq.data[-1]

# ---------------
wave,dw  = 7200,500
dspec = dict(type='template', name='ref/sun', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(ifs, dspec)
spec = spec.subspec(lmin=7000,lmax=7300)
mag = 23
flux = mag2flux(mag, wave)
obs = dict(
    seeing = 0.7,
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
    ima_aperture_type = 'circular_adaptative',
)
wst.set_obs(obs)
moon = 'darksky'

frac_ima,size_ima,nspaxels = wst.get_psf_frac_ima(ifs, flux, spec, moon)
assert size_ima.data[0] > size_ima.data[-1]
assert frac_ima.data.max() < 1
assert frac_ima.data.min() > 0
assert nspaxels[0] > nspaxels[-1]

# ------------------

spec = wst.get_spec(mos, dspec)
spec = spec.subspec(lmin=7000,lmax=7050)

obs = dict(
    seeing = 0.7,
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
)
wst.set_obs(obs)
moon = 'darksky'

frac_ima,size_ima,nspaxels = wst.get_psf_frac_ima(mos, flux, spec, moon)
assert size_ima.data[0] == size_ima.data[-1]
assert frac_ima.data.max() < 1
assert frac_ima.data.min() > 0
assert nspaxels[0] == nspaxels[-1]

# ---------------
wave,dw  = 7200,500
dspec = dict(type='template', name='ref/sun', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(ifs, dspec)
spec = spec.subspec(lmin=7000,lmax=7300)
mag = 23
flux = mag2flux(mag, wave)
obs = dict(
    seeing = 0.7,
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
    ima_aperture_type = 'circular_adaptative',
)
wst.set_obs(obs)
moon = 'darksky'
res = wst.snr_from_source(ifs, flux, None, spec, moon)
sp = res['spec']

#------------------
wave = 8000 # central wavelength A
dspec = dict(type='line', lbda=wave, sigma=2.0, skew=7.0)
spec = wst.get_spec(ifs, dspec)

obs = dict(
    seeing = 0.7,
    airmass = 1,
    ndit = 2, # number of exposures
    dit = 1800, # exposure time in sec of one exposure
    spec_type = 'line',
    spec_range_type = 'adaptative',
    spec_range_kfwhm = 1.5,
    ima_type = 'ps', 
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 7.4,
)
wst.set_obs(obs)
moon = 'darksky'
flux = 5.e-18
kfwhm_spec = wst.optimum_spectral_range(ifs, flux, None, spec, moon)
kfwhm_ima = wst.optimum_circular_aperture(ifs, flux, None, spec, moon)
res = wst.snr_from_source(ifs, flux, None, spec, moon)

snr0 = res['aper']['snr']
res = wst.flux_from_source(ifs, snr0, None, spec, moon)
aper = res['aper']
assert_allclose(aper['snr'],snr0,rtol=0.01)
assert_allclose(aper['flux'],5.e-18,rtol=0.1) # 5.3e-18

# -------------------------------
wave,dw  = 7200,500
dspec = dict(type='template', name='ref/sun', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(ifs, dspec)
spec = spec.subspec(lmin=7000,lmax=7300)
mag = 23
flux = mag2flux(mag, wave)
obs = dict(
    seeing = 0.7,
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'ps',
    ima_aperture_type = 'circular_adaptative',
)
wst.set_obs(obs)
moon = 'darksky'
snr0 = 5
res = wst.flux_from_source(ifs, snr0, None, spec, moon, waves=(7000,7300))
sp = res['spec']
assert_allclose(sp['snr'].mean(lmin=7000, lmax=7300)[0], snr0, rtol=0.01)
