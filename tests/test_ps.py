# Test point source SN computation
from pyetc.wst import WST
from numpy.testing import assert_almost_equal
from mpdaf.obj import flux2mag

etc = WST(log='INFO')

ifs = etc.ifs['blue']
obs = dict(
    ndit = 2, 
    dit = 1800, 
    nb_spectels = 3,
    airmass = 1,
    seeing = 0.7,
    frac_flux = 0.8,
)
etc.set_obs(obs)

frac_flux,nspaxels,rad   = etc.fluxfrac(ifs)
assert (frac_flux > 0) and (frac_flux < 1)
assert nspaxels > 1
assert (rad > 0) and (rad < 2.0)

flux = 1e-18
res = etc.snr_from_psflux(ifs, flux, 'darksky')
assert_almost_equal(res['flux'], flux)
assert res['snr'].data.min() > 0
assert res['snr'].data.max() > res['snr'].data.min()
assert res['snr'].shape[0] == ifs['instrans'].shape[0]
assert res['noise']['tot'].shape[0] == ifs['instrans'].shape[0]

k = 2000
wave = res['snr'].wave.coord(k)
snr0 = res['snr'].data[k]

res3 = etc.psflux_from_snr(ifs, snr0, 'darksky')
assert_almost_equal(res3['flux'].data[k]*1.e18, flux*1.e18)

mag = flux2mag(flux/ifs['dlbda'], 0, wave)[0]
res2 = etc.snr_from_psmag(ifs, mag, 'darksky')
assert_almost_equal(res2['snr'].data[k], snr0)

res4 = etc.psmag_from_snr(ifs, snr0, 'darksky')
assert_almost_equal(res4['mag'].data[k], mag)

mos = etc.moslr['red']
frac_flux,nbspaxels,rad = etc.fluxfrac(mos)
assert (frac_flux > 0) and (frac_flux < 1)

flux = 1e-18
res = etc.snr_from_psflux(mos, flux, 'darksky')
assert_almost_equal(res['flux'], flux)
assert res['snr'].data.min() > 0
assert res['snr'].data.max() > res['snr'].data.min()
assert res['snr'].shape[0] == mos['instrans'].shape[0]
assert res['noise']['tot'].shape[0] == mos['instrans'].shape[0]

k = 2000
wave = res['snr'].wave.coord(k)
mag = flux2mag(flux/mos['dlbda'], 0, wave)[0]
snr0 = res['snr'].data[k]

res2 = etc.snr_from_psmag(mos, mag, 'darksky')
assert_almost_equal(res2['snr'].data[k], res['snr'].data[k])

res3 = etc.psflux_from_snr(mos, snr0, 'darksky')
assert_almost_equal(res3['flux'].data[k]*1.e18, flux*1.e18)

res4 = etc.psmag_from_snr(mos, snr0, 'darksky')
assert_almost_equal(res4['mag'].data[k], mag)
