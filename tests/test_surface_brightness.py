# Test surface brightness SN computation
import numpy as np
from pyetc.wst import WST
from numpy.testing import assert_raises, assert_allclose
from mpdaf.obj import flux2mag,mag2flux

wst = WST(log='DEBUG')

def test_flatcont():
    
    
    ifs = wst.ifs['red']
   
    # -------------------------------------
    obs = dict(
        moon = 'greysky',
        airmass = 1.0,
        ndit = 2, 
        dit = 1800, 
        spec_type = 'cont',
        ima_type = 'sb', 
        ima_area = 1.0 # arcsec**2 only of IFS
    )
    wave = 8000
    dspec = dict(type='flatcont', wave=[wave-10,wave+10])
    spec = wst.get_spec(ifs, dspec)
    mag = 25
    flux = mag2flux(mag, wave)
    wst.set_obs(obs)
    res = wst.snr_from_source(ifs, flux, None, spec)
    sp = res['spec']
    assert sp['snr'].data.min() > 0
    k = sp['snr'].wave.pixel(wave, nearest=True)
    wrange = [wave-10,wave+10]
    krange = sp['snr'].wave.pixel(wrange, nearest=True)
    snr0 = np.mean(sp['snr'].data[krange[0]:krange[1]])
    res = wst.flux_from_source(ifs, snr0, None, spec, snrcomp=dict(method='mean',waves=wrange))
    mag = flux2mag(res['spec']['flux'], 0, wave)[0]
    assert_allclose(res['spec']['snr_mean'], snr0, rtol=0.01) 
    assert_allclose(mag, 25.0, rtol=0.01) 

def test_error_obs():
    ifs = wst.ifs['red']
    wave = 8000
    # --------------------------------
    dspec = dict(type='flatcont', wave=[wave-10,wave+10])
    spec = wst.get_spec(ifs, dspec)
    flux = 1.e-18 
    obs = dict(
        moon = 'greysky',
        airmass = 1.05,
        ndit = 2, 
        dit = 1800, 
        spec_type = 'cont',
        ima_type = 'sb', 
        ima_area = 1.0 # arcsec**2 only of IFS
    )
    wst.set_obs(obs)
    with assert_raises(ValueError):
        wst.snr_from_source(ifs, flux, None, spec)
    
    obs = dict(
        moon = 'greysky',
        ndit = 2, 
        dit = 1800, 
        spec_type = 'cont',
        ima_type = 'sb', 
        ima_area = 1.0 # arcsec**2 only of IFS
    )
    wst.set_obs(obs)
    with assert_raises(KeyError):
        wst.snr_from_source(ifs, flux, None, spec)

# -------------------------------------

def test_flux_from_source():
   
    mos = wst.moslr['red']    
    
    obs = dict(
        moon = 'greysky',
        airmass = 1.0,
        ndit = 2, 
        dit = 1800, 
        spec_type = 'cont',
        ima_type = 'sb', 
    )
    wave = 8000
    dspec = dict(type='flatcont', wave=[wave-10,wave+10])
    spec = wst.get_spec(mos, dspec)
    mag = 25
    flux = mag2flux(mag, wave)
    wst.set_obs(obs)
    res = wst.snr_from_source(mos, flux, None, spec)
    sp = res['spec']
    assert sp['snr'].data.min() > 0
    k = sp['snr'].wave.pixel(wave, nearest=True)
    wrange = [wave-10,wave+10]
    krange = sp['snr'].wave.pixel(wrange, nearest=True)
    snr0 = np.mean(sp['snr'].data[krange[0]:krange[1]])
    res = wst.flux_from_source(mos, snr0, None, spec, snrcomp=dict(method='mean',waves=wrange))
    mag = flux2mag(res['spec']['flux'], 0, wave)[0]
    assert_allclose(res['spec']['snr_mean'], snr0, rtol=0.01) 
    assert_allclose(mag, 25.0, rtol=0.01) 
    

# -------------------------------------
def test_line():
    ifs = wst.ifs['red']
    
    obs = dict(
        moon = 'greysky',
        airmass = 1.0,
        ndit = 2, 
        dit = 1800, 
        spec_type = 'line',
        spec_range_type = 'adaptative',
        spec_range_kfwhm = 3,    
        ima_type = 'sb', 
        ima_area = 5.0 # arcsec**2 only of IFS   
    )
    wave = 8000
    dspec = dict(type='line', lbda=wave, sigma=4.0, skew=7.0)
    spec = wst.get_spec(ifs, dspec)
    flux = 5.e-18
    wst.set_obs(obs)
    res = wst.snr_from_source(ifs, flux, None, spec)
    sp = res['spec']
    assert sp['snr_mean'] > 0
