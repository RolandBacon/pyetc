# Test setup
from pyetc.wst import WST
from pyetc.etc import compute_sky

etc = WST(log='INFO')
assert hasattr(etc,'ifs'), 'attribute ifs missing'
assert hasattr(etc,'moslr'), 'attribute moslr missing'
assert hasattr(etc,'tel'), 'attribute tel missing'
assert etc.name == 'WST', 'WST expected as name'
for chan in etc.ifs['channels']:
    assert etc.ifs[chan]['dlbda'] > 0, 'dlbda must be > 0'
    assert etc.ifs[chan]['lbda2'] > etc.ifs[chan]['lbda1'], 'lbda2 muste be > lbda1'

# test sky computation    
ifs = etc.ifs['blue']
tab = compute_sky(ifs['lbda1'], ifs['lbda2'], ifs['dlbda'], 2.5, 'greysky')
assert len(tab) > 0, 'resulting sky table has zero length'
assert 'lam' in tab.columns, 'lam column not found'

# test get spectral resolution
res = etc.get_spectral_resolution(ifs)
assert res.shape[0] == ifs['instrans'].shape[0]

