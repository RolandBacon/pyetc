import logging
from astropy import constants
import astropy.units as u
C_cgs = constants.c.cgs.value
H_cgs = constants.h.cgs.value
C_kms = constants.c.to(u.km/u.s).value
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord, Cube, WCS
from mpdaf.obj import gauss_image, mag2flux, flux2mag, moffat_image
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
from scipy.special import erf
from scipy.optimize import root_scalar


CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/WST')

class ETC:
    
    def __init__(self, log=logging.INFO):
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        
        
    def info(self):
        self.logger.info('%s ETC version: %s', self.name, self.version)                        
              
    def set_obs(self, obs):
        obs['totexp'] = obs['ndit']*obs['dit']/3600.0 # tot integ time in hours
        if 'nradius_spaxels' in obs.keys():
            nb_spaxels = (2*obs['nradius_spaxels']+1)**2
            if 'nb_spaxels' in obs.keys():
                if (obs['nb_spaxels'] != nb_spaxels):
                    raise ValueError('Incompatible nb_spaxels and nradius_spaxels in obs')
            obs['nb_spaxels'] = nb_spaxels
        self.obs = obs
        
    def snr_from_sb(self, sbflux, moon, channel):
        """ compute SNR(lambda) for a source with a given surface brightness summed over nb_spaxels
            for an emsission line integrated over nb_spectels 
            sbflux erg/s/cm2/arcsec2
            moon condition [darksky, greysky]
            channel [blue, red]"""
        self._check_moon(moon)
        self._check_channel(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']
        flux_source = sbflux * dl / area
        res = self._snr_from_flux(flux_source, moon, channel)
        res['dl'] = dl
        res['area'] = area
        res['sbflux'] = sbflux
        res['moon'] = moon
        res['channel'] = channel
        return res
    
    def snr_from_esmag(self, mag, moon, channel):
        """ compute SNR(lambda) for an extended source with a given surface AB magnitude summed over nb_spaxels
            the flux is summed over nb_spectels 
            mag AB magnitude
            moon condition [darksky, greysky]
            channel [blue, red]"""
        self._check_moon(moon)
        self._check_channel(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']
        sbflux = mag2flux(mag, ifs['atmtrans'].wave.coord())
        flux_source = sbflux * dl / area
        res = self._snr_from_flux(flux_source, moon, channel)
        res['dl'] = dl
        res['area'] = area
        res['mag'] = mag
        res['moon'] = moon
        res['channel'] = channel        
        return res 
    
    def get_source(self, lbda, ima_fwhm, ima_beta, spec_sigma, spec_skew, channel, 
                   ima_kfwhm=3, spec_kfwhm=3):
        """ return the source of total flux in erg/s/cm2
        composed by an emission line (skewed gaussian, lbda, spec_sigma, spec_skew)
        and a moffat image (ima_fwhm, ima_skew) """
        ifs = self.ifs[channel]
        ima = moffat(ifs['spaxel_size'], ima_fwhm, ima_beta, kfwhm=ima_kfwhm)
        dl = spec_kfwhm*10*spec_sigma
        wave = np.linspace(lbda-dl,lbda+dl)
        f = asymgauss(1.0, lbda, spec_sigma, spec_skew, wave)
        sp = Spectrum(data=f, wave=WaveCoord(cdelt=(wave[1]-wave[0]), crval=wave[0]))
        l0,l1,l2 = fwhm_asymgauss(sp.wave.coord(),sp.data)
        dl1,dl2 = l1-l0,l2-l0
        l1,l2 = l0+spec_kfwhm*dl1,l0+spec_kfwhm*dl2
        temp = ifs['instrans'].subspec(lmin=l1, lmax=l2)
        ssp = sp.subspec(lmin=l1, lmax=l2)
        frac = ssp.data.sum()/sp.data.sum()
        rspec = sp.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'), 
                              shape=temp.shape[0])
        rspec.data *= frac/rspec.data.sum()
        return ima,rspec
              
 
    def _snr_from_spec(self, spec, moon, channel):
        """ spec input spectrum in erg/s/cm2/spectel
            return snr spectrum """
        ifs = self.ifs[channel]
        obs = self.obs
        # truncate instrans and sky to the wvalength limit of the input spectrum
        ifs_sky = ifs[moon].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        ifs_ins = ifs['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        ifs_atm = ifs['atmtrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        # rebin the input spectrum to the IFS channel 
        rspec = spec.resample(ifs_sky.get_step(unit='Angstrom'), start=ifs_sky.get_start(unit='Angstrom'), 
                              shape=ifs_sky.shape[0])
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = rspec.get_step(unit='Angstrom')
        w = rspec.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ifs_atm.data**obs['airmass'])
        Kt =  ifs_ins * a
        nph_source = obs['dit'] * obs['ndit'] * Kt * rspec.data # number of photons received from the source
        nph_sky = ifs_sky * ifs_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = obs['nb_spaxels'] # for 1 spectel
        ron_noise = np.sqrt(ifs['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ifs['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = np.sqrt(nph_source)
        tot_noise = source_noise.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2) 
        snr = nph_source / tot_noise 
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, flux_source=rspec, nph_source=nph_source, nph_sky=nph_sky)
        return res         
  
    def snr_from_cube(self, cube, moon, channel):
        ifs = self.ifs[channel]
        obs = self.obs
        # truncate instrans and sky to the wavelength limit of the input spectrum
        ifs_sky = ifs[moon].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ifs_ins = ifs['instrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ifs_atm = ifs['atmtrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area
        dl = cube.wave.get_step(unit='Angstrom')
        w = cube.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ifs_atm.data**obs['airmass'])
        Kt =  ifs_ins * a
        nph_source = cube.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt[:,np.newaxis,np.newaxis].data * cube.data # number of photons received from the source
        nph_sky = ifs_sky * ifs_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ifs['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ifs['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = cube.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = cube.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data[:,np.newaxis,np.newaxis]**2 + source_noise.data**2) 
        snr = cube.copy()
        snr.data = nph_source.data / tot_noise.data 
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr_cube=snr, flux_source=cube, nph_source=nph_source, nph_sky=nph_sky)
        return res         
        
    
    def _snr_from_flux(self, flux_source, moon, channel):
        """ flux_source in erg/s/cm2 """
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']
        w = ifs['atmtrans'].wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ifs['atmtrans'].data**obs['airmass'])
        Kt =  ifs['instrans'] * a
        nph_source = obs['dit'] * obs['ndit'] * Kt * flux_source # number of photons received from the source
        nph_sky = ifs[moon] * ifs['instrans'] * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = obs['nb_spaxels']*obs['nb_spectels']
        ron_noise = np.sqrt(ifs['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ifs['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = np.sqrt(nph_source)
        tot_noise = source_noise.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2) 
        snr = nph_source / tot_noise 
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, flux_source=flux_source, nph_source=nph_source, nph_sky=nph_sky)
        return res 
    
    def sb_from_snr(self, snr, moon, channel):
        """ compute surface brightness flux(lambda) in erg/s/cm2/arcsec2 for an extended source with a given snr
            the signal is summed over nb_spaxels and the emsission line integrated over nb_spectels 
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """        
        self._check_moon(moon)
        self._check_channel(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']        
        res = self._flux_from_snr(snr, moon, channel)
        res['sbflux'] = res['flux_source'] * dl / area 
        res['moon'] = moon
        res['channel'] = channel
        res['snr'] = snr
        res['dl'] = dl
        res['area'] = area        
        return res 

    def esmag_from_snr(self, snr, moon, channel):
        """ compute surface brightness AB mag(lambda) for an extended source with a given snr
            the signal is summed over nb_spaxels and the emsission line integrated over nb_spectels 
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """        
        self._check_moon(moon)
        self._check_channel(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']        
        res = self._flux_from_snr(snr, moon, channel)
        sbflux = res['flux_source']/(dl*area)
        res['mag'] = sbflux.to_abmag()
        res['moon'] = moon
        res['channel'] = channel
        res['snr'] = snr
        res['dl'] = dl
        res['area'] = area        
        return res 
    
            
    def _flux_from_snr(self, snr, moon, channel):   
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']
        w = ifs['atmtrans'].wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ifs['atmtrans'].data**obs['airmass'])
        Kt =  ifs['instrans'] * a
        nph_sky = ifs[moon] * ifs['instrans'] * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = obs['nb_spaxels']*obs['nb_spectels']
        ron_noise2 = ifs['ron']**2*nb_voxels*obs['ndit']
        dark_noise2 = ifs['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600
        sky_noise2 = nph_sky
        N2 = ron_noise2 + dark_noise2 + sky_noise2
        nph_source = 0.5*(snr**2 + snr*np.sqrt(snr**2+4*N2)) # nb of source photons
        flux_source = nph_source/(obs['ndit']*obs['dit'] * Kt) # source flux in erg/s/cm2
        ron_noise = np.sqrt(ron_noise2)
        dark_noise = np.sqrt(dark_noise2)
        sky_noise = np.sqrt(sky_noise2)        
        source_noise = np.sqrt(nph_source)
        tot_noise = source_noise.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2) 
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, flux_source=flux_source, nph_source=nph_source, nph_sky=nph_sky)        
        return res 
    
    def fluxfrac(self, channel):
        """ Compute the flux fraction into a given squared aperture
            the psf is assumed to be gaussian with FHM=seeing and independant of wavelength
        """
        ifs = self.ifs[channel]
        seeing_pix = self.obs['seeing']/ifs['spaxel_size']
        nrad = self.obs['nradius_spaxels']
        nspa = 2*nrad + 1
        hdim = max(int(2*seeing_pix+0.5), nspa)
        psf = gauss_image(shape=(2*hdim+1,2*hdim+1), fwhm=(seeing_pix,seeing_pix), unit_center=None, unit_fwhm=None)
        spsf = psf[hdim-nrad:hdim+nrad+1,hdim-nrad:hdim+nrad+1]
        frac_flux = spsf.data.sum()
        return frac_flux
    
    def snr_from_psflux(self, flux, moon, channel):
        """ compute snr(lambda) for a point source with a given flux
            the emission line is integrated over nb_spectels 
            flux in erg/s/cm2
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        self._check_channel(channel)
        frac_flux = self.fluxfrac(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        dl = obs['nb_spectels'] * ifs['dlbda']
        flux_source = flux * dl * frac_flux
        res = self._snr_from_flux(flux_source, moon, channel)
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['flux'] = flux
        res['moon'] = moon
        res['channel'] = channel            
        return res 
    
    def flux_from_source(self, snr, ima, spec, moon, channel, bracket=(0.1,1000)):
        """ compute flux for a given snr and source defined
            by flux x image x spectra
            flux in erg/s/cm2
            moon condition [darksky, greysky]
            channel [blue, red]
        """ 
        res0 = root_scalar(self.fun, args=(snr, ima, spec, moon, channel), method='brentq', 
                              bracket=bracket, rtol=1.e-3, maxiter=100)
        flux = res0.root*1.e-20
        res = self.snr_from_source(flux, ima, spec, moon, channel)
        res['aper']['flux'] = flux
        return res
    
    def fun(self, flux, snr0, ima, spec, moon, channel):
        res = self.snr_from_source(flux*1.e-20, ima, spec, moon, channel)
        snr = res['aper']['snr']
        #print(flux,snr)
        return snr-snr0
        
    
        
    def snr_from_source(self, flux, ima, spec, moon, channel):
        """ compute snr cube and on an aperture for a source defined
            by flux x image x spectra
            flux in erg/s/cm2
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        self._check_channel(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        dl = ifs['dlbda']
        cube = flux * ima * spec
        res = self.snr_from_cube(cube, moon, channel)
        # sum over an aperture spatial and spectral
        data = res['nph_source']
        sky = res['nph_sky']
        # find centered spatial aperture
        temp = data.sum(axis=0)
        r = temp.peak()
        p,q = int(r['p']+0.5),int(r['q']+0.5)
        hpix = obs['nradius_spaxels']
        stemp = temp.data[p-hpix:p+hpix+1,q-hpix:q+hpix+1] 
        frac_ima = stemp.data.sum()/temp.data.sum()
        nvoxels = np.prod(data[:,p-hpix:p+hpix+1,q-hpix:q+hpix+1].shape)
        nspaxels = (2*hpix+1)**2        
        # truncate to the spatial aperture
        sky = res['nph_sky']
        vartot = res['noise']['tot'].data**2
        varsky = res['noise']['sky'].data**2 
        sdata = data[:,p-hpix:p+hpix+1,q-hpix:q+hpix+1]
        svartot = vartot[:,p-hpix:p+hpix+1,q-hpix:q+hpix+1]
        # compute snr
        snr = sdata.sum()/np.sqrt(svartot.sum())
        # fraction of flux recovered
        frac_flux = cube[:,p-hpix:p+hpix+1,q-hpix:q+hpix+1].sum()/flux
        # save data for the aperture
        res['aper'] = dict(frac_flux=frac_flux,
                           frac_ima=frac_ima,
                           frac_spec=frac_flux/frac_ima,
                           snr=snr,
                           nph_source=sdata.sum(),
                           nph_sky=nspaxels*sky.data.sum(),
                           ron=np.sqrt(nvoxels)*res['noise']['ron'],
                           dark=np.sqrt(nvoxels)*res['noise']['dark'],
                           sky_noise=np.sqrt(nspaxels*sky.data.sum()),
                           source_noise=np.sqrt(sdata.sum()),
                           tot_noise=np.sqrt(svartot.sum()),
                           )
        detvar = res['aper']['ron']**2+res['aper']['dark']**2
        skyvar = res['aper']['sky_noise']**2+res['aper']['source_noise']**2
        res['aper']['detnoise'] = np.sqrt(detvar/skyvar)
        res['dl'] = dl
        res['flux'] = flux
        res['moon'] = moon
        res['channel'] = channel            
        return res     
    
    def snr_from_psmag(self, mag, moon, channel):
        """ compute snr(lambda) for a point source with a given AB magnitude
            the emission line is integrated over nb_spectels 
            mag AB magnitude
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        self._check_channel(channel)
        frac_flux = self.fluxfrac(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        dl = obs['nb_spectels'] * ifs['dlbda']
        flux = mag2flux(mag, ifs['atmtrans'].wave.coord())
        flux_source = flux * dl * frac_flux
        res = self._snr_from_flux(flux_source, moon, channel)
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['mag'] = mag
        res['moon'] = moon
        res['channel'] = channel            
        return res     
    
    def psflux_from_snr(self, snr, moon, channel):
        """ compute flux(lambda) in erg/s/cm2 for a point source with a given snr
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        self._check_channel(channel)
        frac_flux = self.fluxfrac(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        dl = obs['nb_spectels'] * ifs['dlbda']
        res = self._flux_from_snr(snr, moon, channel)
        res['flux'] = res['flux_source'] * dl / frac_flux          
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['snr'] = snr
        res['moon'] = moon
        res['channel'] = channel             
        return res     
    
    def psmag_from_snr(self, snr, moon, channel):
        """ compute AB mag(lambda) for a point source with a given snr
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        self._check_channel(channel)
        frac_flux = self.fluxfrac(channel)
        ifs = self.ifs[channel]
        obs = self.obs
        dl = obs['nb_spectels'] * ifs['dlbda']
        res = self._flux_from_snr(snr, moon, channel)
        flux = res['flux_source'] / (frac_flux  * dl)
        res['mag'] = flux.to_abmag()       
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['snr'] = snr
        res['moon'] = moon
        res['channel'] = channel        
        return res         
        
        
    def _check_moon(self, moon):
        if moon not in self.ifs['skys']:
            raise ValueError('Unknown moon') 
    
    def _check_channel(self, channel):
        if channel not in self.ifs['channels']:
            raise ValueError('Unknown channel')
        
        
def asymgauss(ftot, l0, sigma, skew, wave):
    """ compute asymetric gaussian 
    ftot: total flux
    l0: peak wavelength in A
    sigma: sigma in A
    skew: skew parameter 0=gauss
    wave: array of wavelength array (A)
    return: array fo flux
    """
    dl = wave - l0
    g = np.exp(-dl**2/(2*sigma**2))
    f = 1 + erf(skew*dl/(1.4142135623730951*sigma))
    h = f*g
    h = ftot * h/h.sum()
    return h 

def vdisp2sigma(vdisp, l0):
    """ vdisp in km/s
    l0 in A
    return sigma in A
    """
    return vdisp*l0/C_kms

def sigma2vdisp(sigma, l0):
    """ sigma in A
    l0 in A
    return vdisp in km/s
    """
    return sigma*C_kms/l0

def fwhm_asymgauss(lbda, flux):
    g = flux/flux.max()    
    kmax = g.argmax()
    l0 = lbda[kmax]
    l1 = None
    for k in range(kmax,0,-1):
        if g[k] < 0.5:
            l1 = np.interp(0.5, [g[k],g[k+1]],[lbda[k],lbda[k+1]])
            break
    if l1 is None:
        return None
    l2 = None
    for k in range(kmax,len(lbda),1):
        if g[k] < 0.5:
            l2 = np.interp(0.5, [g[k],g[k-1]],[lbda[k],lbda[k-1]])
            break
    if l2 is None:
        return None 
    return l0,l1,l2

def moffat(samp, fwhm, beta, kfwhm=2):
    ns = int((kfwhm*fwhm/samp+1)/2)*2 + 1
    ima = moffat_image(fwhm=(fwhm/samp,fwhm/samp), n=beta, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.data /= ima.data.sum()
    return ima

