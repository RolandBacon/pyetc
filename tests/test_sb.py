# Test surface brightness SN computation
from pyetc.wst import WST
from numpy.testing import assert_almost_equal
from mpdaf.obj import flux2mag

etc = WST(log='INFO')

ifs = etc.ifs['blue']
obs = dict(
    ndit = 2, 
    dit = 1800,
    nb_spaxels = 4 * 4, 
    nb_spectels = 3,
    airmass = 1,
)
etc.set_obs(obs)
sbflux = 1e-18 
res = etc.snr_from_sb(ifs, sbflux, 'darksky')
assert res['area'] == 1.0, 'area != 1.0'
assert_almost_equal(res['sbflux'], sbflux)
assert res['snr'].data.min() > 0
assert res['snr'].data.max() > res['snr'].data.min()
assert res['snr'].shape[0] == ifs['instrans'].shape[0]
assert res['noise']['tot'].shape[0] == ifs['instrans'].shape[0]

k = 2000
wave = res['snr'].wave.coord(k)
mag = flux2mag(sbflux, 0, wave)[0]
snr0 = res['snr'].data[k]
res2 = etc.snr_from_esmag(ifs, mag, 'darksky')
assert_almost_equal(res2['snr'].data[k], res['snr'].data[k])

res3 = etc.sb_from_snr(ifs, snr0, 'darksky')
assert_almost_equal(res3['sbflux'].data[k]*1.e18, sbflux*1.e18, decimal=3)

res4 = etc.esmag_from_snr(ifs, snr0, 'darksky')
assert_almost_equal(res4['mag'].data[k], mag)
