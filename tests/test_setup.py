# Test setup
from pyetc.wst import WST
from pyetc.etc import compute_sky, update_skytables

etc = WST(log='DEBUG')

def test_setup():
       
    assert hasattr(etc,'ifs'), 'attribute ifs missing'
    assert hasattr(etc,'moslr'), 'attribute moslr missing'
    assert hasattr(etc,'tel'), 'attribute tel missing'
    assert etc.name == 'WST', 'WST expected as name'
    for chan in etc.ifs['channels']:
        assert etc.ifs[chan]['dlbda'] > 0, 'dlbda must be > 0'
        assert etc.ifs[chan]['lbda2'] > etc.ifs[chan]['lbda1'], 'lbda2 must be > lbda1'
        
    # test print info
    etc.info()

def test_sky():    
    # test sky computation    
    ifs = etc.ifs['blue']
    tab = compute_sky(ifs['lbda1'], ifs['lbda2'], ifs['dlbda'], 2.5, 'greysky', 1.5)
    assert len(tab) > 0, 'resulting sky table has zero length'
    assert 'lam' in tab.columns, 'lam column not found'
    
    # update of sky files
    ifs = etc.ifs['red']
    update_skytables(etc.logger, ifs, ['brightsky'], [1.0], etc.refdir, overwrite=False, debug=True)
    

def test_spectral_resolution():     
    # test get spectral resolution
    ifs = etc.ifs['blue']
    res = etc.get_spectral_resolution(ifs)
    assert res.shape[0] == ifs['instrans'].shape[0]
    




