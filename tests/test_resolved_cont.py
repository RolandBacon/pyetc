# Test source resolved and cont
from pyetc.wst import WST
import numpy as np
from mpdaf.obj import mag2flux,flux2mag
from numpy.testing import assert_almost_equal, assert_allclose

wst = WST(log='DEBUG')
ifs = wst.ifs['red']
mos = wst.moslr['red']

# ---------------
wave = 7000
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = wst.get_spec(ifs, dspec)
assert_allclose(spec.data, np.ones(spec.shape))

# ---------------
wave = 7500
dw = 500
dspec = dict(type='template', name='kc96/starb1', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(ifs, dspec)
vmean = spec.mean(lmin=wave-dw/2,lmax=wave+dw/2)[0]
assert_allclose(vmean, 1.0, rtol=1.e-3)
spec2 = wst.get_spec(wst.ifs['blue'], dspec)

# --------------
wave = 7500
dw = 500
dspec = dict(type='template', name='kc96/starb1', 
             wave_center=wave, wave_width=dw)
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=0.8, beta=2.5)
ima = wst.get_ima(ifs, dima, uneven=0)
obs = dict(
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'resolved',
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 7,
)
mag = 24
flux = mag2flux(mag, wave)
moon = 'greysky'
wst.set_obs(obs)
kfwhm = wst.optimum_circular_aperture(ifs, flux, ima, spec, moon, lrange=[wave-50,wave+50])
assert kfwhm > 0
res = wst.snr_from_source(ifs, flux, ima, spec, moon)
sp = res['spec']
assert sp['snr'].data.min() > 0

# --------------
wave = 7000
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = wst.get_spec(ifs, dspec)
dima = dict(type='moffat', fwhm=0.8, beta=2.5)
ima = wst.get_ima(ifs, dima, uneven=0)

obs = dict(
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'resolved',
    ima_aperture_type = 'circular_adaptative',
    ima_kfwhm = 5,
)
mag = 24
flux = mag2flux(mag, wave)
moon = 'greysky'
wst.set_obs(obs)
res = wst.snr_from_source(ifs, flux, ima, spec, moon)
sp = res['spec']
assert sp['snr'].data.min() > 0
k = sp['snr'].wave.pixel(wave, nearest=True)

wrange = [wave-5,wave+5]
krange = sp['snr'].wave.pixel(wrange, nearest=True)
snr0 = np.mean(sp['snr'].data[krange[0]:krange[1]])
res = wst.flux_from_source(ifs, snr0, ima, spec, moon, waves=wrange)
mag = flux2mag(res['flux'], 0, wave)[0]
assert_allclose(res['spec']['snr_mean'], snr0, rtol=0.01) 
assert_allclose(mag, 24.0, rtol=0.01) 

# -------------------------------------
obs = dict(
    airmass = 1.0,
    ndit = 2, 
    dit = 1800, 
    spec_type = 'cont',
    ima_type = 'resolved',
)
wave = 8000
dspec = dict(type='flatcont', wave=[wave-10,wave+10])
spec = wst.get_spec(mos, dspec)
dima = dict(type='moffat', fwhm=0.8, beta=2.5)
ima = wst.get_ima(mos, dima, uneven=0)
mag = 23
flux = mag2flux(mag, wave)
moon = 'greysky'
wst.set_obs(obs)
res = wst.snr_from_source(mos, flux, ima, spec, moon)
sp = res['spec']
assert sp['snr'].data.min() > 0
k = sp['snr'].wave.pixel(wave, nearest=True)

wrange = [wave-1,wave+1]
krange = sp['snr'].wave.pixel(wrange, nearest=True)
snr0 = np.mean(sp['snr'].data[krange[0]:krange[1]])
res = wst.flux_from_source(mos, snr0, ima, spec, moon, waves=wrange)
mag = flux2mag(res['flux'], 0, wave)[0]
assert_allclose(res['spec']['snr_mean'], snr0, rtol=0.01) 
assert_allclose(mag, 23.0, rtol=0.01) 

