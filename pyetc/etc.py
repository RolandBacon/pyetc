import logging
from astropy import constants
import astropy.units as u
C = constants.c.cgs.value
H = constants.h.cgs.value
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord
from mpdaf.obj import gauss_image, mag2flux, flux2mag
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging


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
        
    
    def _snr_from_flux(self, flux_source, moon, channel):
        """ flux_source in erg/s/cm2 """
        ifs = self.ifs[channel]
        obs = self.obs
        spaxel_area = ifs['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ifs['dlbda']
        w = ifs['atmtrans'].wave.coord() # wavelength in A
        a = (w*1.e-8/(H*C)) * (self.tel['area']*1.e4) * (ifs['atmtrans'].data**obs['airmass'])
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
        a = (w*1.e-8/(H*C)) * (self.tel['area']*1.e4) * (ifs['atmtrans'].data**obs['airmass'])
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
