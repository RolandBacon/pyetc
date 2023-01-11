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
from skycalc_cli.skycalc import SkyModel
from skycalc_cli.skycalc_cli import fixObservatory
from io import BytesIO


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
        
    def get_spectral_resolution(self, ins):
        """ return spectral resolution
        ins instrument (eg self.ifs['blue'] or self.moslr['red']) 
        """
        lsf = ins['lsfpix']*ins['dlbda']
        wave = ins['wave'].coord()
        res = wave/lsf
        return res
        
        
    # ------ Surface Brightness ------------
        
    def snr_from_sb(self, ins, sbflux, moon):
        """ compute SNR(lambda) for a source with a given surface brightness summed over nb_spaxels
            for an emsission line integrated over nb_spectels 
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            sbflux erg/s/cm2/arcsec2
            moon condition [darksky, greysky]
            """
        self._check_moon(moon)
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ins['dlbda']
        flux_source = sbflux * dl / area
        res = self._snr_from_flux(ins, flux_source, moon, obs['nb_spaxels'], obs['nb_spectels'])
        res['dl'] = dl
        res['area'] = area
        res['sbflux'] = sbflux
        res['moon'] = moon
        return res
    
    
    def snr_from_esmag(self, ins, mag, moon):
        """ compute SNR(lambda) for an extended source with a given surface AB magnitude summed over nb_spaxels
            the flux is summed over nb_spectels 
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            mag AB magnitude
            moon condition [darksky, greysky]
            """
        self._check_moon(moon)
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ins['dlbda']
        sbflux = mag2flux(mag, ins['atmtrans'].wave.coord())
        flux_source = sbflux * dl / area
        res = self._snr_from_flux(ins, flux_source, moon, obs['nb_spaxels'], obs['nb_spectels'])
        res['dl'] = dl
        res['area'] = area
        res['mag'] = mag
        res['moon'] = moon      
        return res 
              
    def sb_from_snr(self, ins, snr, moon):
        """ compute surface brightness flux(lambda) in erg/s/cm2/arcsec2 for an extended source with a given snr
            the signal is summed over nb_spaxels and the emsission line integrated over nb_spectels
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """        
        self._check_moon(moon)
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ins['dlbda']        
        res = self._flux_from_snr(ins, snr, moon, obs['nb_spaxels'], obs['nb_spectels'])
        res['sbflux'] = res['flux_source']/(dl*area)
        res['moon'] = moon
        res['snr'] = snr
        res['dl'] = dl
        res['area'] = area        
        return res 

    def esmag_from_snr(self, ins, snr, moon):
        """ compute surface brightness AB mag(lambda) for an extended source with a given snr
            the signal is summed over nb_spaxels and the emsission line integrated over nb_spectels 
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            snr S/N 
            moon condition [darksky, greysky]
        """        
        self._check_moon(moon)
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * obs['nb_spaxels']
        dl = obs['nb_spectels'] * ins['dlbda']        
        res = self._flux_from_snr(ins, snr, moon, obs['nb_spaxels'], obs['nb_spectels'])
        sbflux = res['flux_source']/(dl*area)
        res['mag'] = sbflux.to_abmag()
        res['moon'] = moon
        res['snr'] = snr
        res['dl'] = dl
        res['area'] = area        
        return res 
    
    def _snr_from_flux(self, ins, flux_source, moon, nb_spaxels, nb_spectels):
        """ flux_source in erg/s/cm2 """
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * nb_spaxels
        dl = nb_spectels * ins['dlbda']
        w = ins['atmtrans'].wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins['atmtrans'].data**obs['airmass'])
        Kt =  ins['instrans'] * a
        nph_source = obs['dit'] * obs['ndit'] * Kt * flux_source # number of photons received from the source
        nph_sky = ins[moon] * ins['instrans'] * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = nb_spaxels*nb_spectels
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = np.sqrt(nph_source)
        tot_noise = source_noise.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2) 
        snr = nph_source / tot_noise
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, 
                                tot=tot_noise), 
                   snr=snr, flux_source=flux_source, nph_source=nph_source, nph_sky=nph_sky,
                   nb_spaxels=nb_spaxels, nb_spectels=nb_spectels, nb_voxels=nb_voxels)
        return res 
            
    def _flux_from_snr(self, ins, snr, moon, nb_spaxels, nb_spectels):   
        obs = self.obs
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area * nb_spaxels       
        dl = nb_spectels * ins['dlbda']
        w = ins['atmtrans'].wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins['atmtrans'].data**obs['airmass'])
        Kt =  ins['instrans'] * a
        nph_sky = ins[moon] * ins['instrans'] * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = nb_spaxels*nb_spectels
        ron_noise2 = ins['ron']**2*nb_voxels*obs['ndit']
        dark_noise2 = ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600
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
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, 
                                tot=tot_noise), 
                   snr=snr, flux_source=flux_source, nph_source=nph_source, nph_sky=nph_sky)        
        return res     
    
    #------- point source ----------
    def snr_from_psflux(self, ins, flux, moon):
        """ compute snr(lambda) for a point source with a given flux
            the emission line is integrated over nb_spectels 
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            flux in erg/s/cm2
            moon condition [darksky, greysky]
        """
        self._check_moon(moon)
        frac_flux,nspaxels,rad = self.fluxfrac(ins)
        obs = self.obs
        dl = obs['nb_spectels'] * ins['dlbda']
        flux_source = flux * frac_flux
        res = self._snr_from_flux(ins, flux_source, moon, nspaxels, obs['nb_spectels'])
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['flux'] = flux
        res['moon'] = moon 
        res['aperture'] = 2*rad 
        return res 
    
    def fluxfrac(self, ins):
        """ Compute the flux fraction into a circular aperture
            for ins['type'] = IFS: the radius is computed to achieve the selected frac_flux
            for ins['type'] = MOS: the radius is fixed 
            the psf is assumed to be gaussian with FHM=seeing and independant of wavelength
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            return frac_flux (the flux fraction captured in the aperture), nspaxels (the corresponding number of spaxels,
            rad the aperture radius)
        """ 
        sig2 = (self.obs['seeing']/2.355)**2
        if ins['type'] == 'IFS':
            frac_flux = self.obs['frac_flux']
            rad = np.sqrt(-2*sig2*np.log(2*np.pi*sig2*(1-frac_flux)))          
        elif ins['type'] == 'MOS':
            rad = ins['aperture']/2
            frac_flux = 1 - np.exp(-rad**2/(2*sig2))/(2*np.pi*sig2) 
        else:
            raise ValueError(f"unknown instrument type {ins['type']}")
        nspaxels = int(np.pi*rad**2/ins['spaxel_size']**2 + 0.5)
        return frac_flux,nspaxels,rad  
 
    def snr_from_psmag(self, ins, mag, moon):
        """ compute snr(lambda) for a point source with a given AB magnitude
            the emission line is integrated over nb_spectels 
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            mag AB magnitude
            moon condition [darksky, greysky]
        """
        self._check_moon(moon)
        frac_flux,nspaxels,rad = self.fluxfrac(ins)
        obs = self.obs
        dl = ins['dlbda']
        flux = mag2flux(mag, ins['atmtrans'].wave.coord()) # return flux/A
        flux_source = flux * dl * frac_flux
        res = self._snr_from_flux(ins, flux_source, moon, nspaxels, obs['nb_spectels'])
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['mag'] = mag
        res['moon'] = moon  
        res['aperture'] = 2*rad 
        return res     
    
    def psflux_from_snr(self, ins, snr, moon):
        """ compute flux(lambda) in erg/s/cm2 for a point source with a given snr
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            snr S/N 
            moon condition [darksky, greysky]
            channel [blue, red]
        """
        self._check_moon(moon)
        frac_flux,nspaxels,rad = self.fluxfrac(ins)
        obs = self.obs
        dl = obs['nb_spectels'] * ins['dlbda']
        res = self._flux_from_snr(ins, snr, moon, nspaxels, obs['nb_spectels'])
        res['flux'] = res['flux_source']/frac_flux          
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['snr'] = snr
        res['moon'] = moon
        res['aperture'] = 2*rad 
        return res     
    
    def psmag_from_snr(self, ins, snr, moon):
        """ compute AB mag(lambda) for a point source with a given snr
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            snr S/N 
            moon condition [darksky, greysky]
        """
        self._check_moon(moon)
        frac_flux,nspaxels,rad = self.fluxfrac(ins)
        obs = self.obs
        dl = ins['dlbda']
        res = self._flux_from_snr(ins, snr, moon, nspaxels, obs['nb_spectels'])
        flux = res['flux_source'] / (frac_flux * dl)
        res['mag'] = flux.to_abmag()       
        res['frac_flux'] = frac_flux
        res['dl'] = dl
        res['seeing'] = self.obs['seeing']
        res['snr'] = snr
        res['moon'] = moon
        res['aperture'] = 2*rad 
        return res  
    
    # ---------- source defined as image x spectrum -------------
    
    def get_source(self, ins, lbda, ima_fwhm, ima_beta, spec_sigma, spec_skew, 
                   ima_kfwhm=5, spec_kfwhm=5, oversamp=5):
        """ return the source of total flux in erg/s/cm2
        ins instrument (eg self.ifs['blue'] or self.moslr['red'])
        composed by an emission line (skewed gaussian, lbda, spec_sigma, spec_skew)
        and a moffat image (ima_fwhm, ima_skew) """
        ima = moffat(ins['spaxel_size'], ima_fwhm, ima_beta, kfwhm=ima_kfwhm, oversamp=oversamp)
        ima.oversamp = oversamp
        
        dl = spec_kfwhm*10*spec_sigma
        lstep = ins['instrans'].get_step()
        if spec_skew == 0:
            l0 = lbda
        else:
            l0 = peakwave_asymgauss(lbda, spec_sigma, spec_skew)
        wave = np.arange(lbda-dl,lbda+dl,lstep/oversamp)
        f = asymgauss(1.0, l0, spec_sigma, spec_skew, wave)
        sp = Spectrum(data=f, wave=WaveCoord(cdelt=(wave[1]-wave[0]), crval=wave[0]))
        l0,l1,l2 = fwhm_asymgauss(sp.wave.coord(),sp.data)
        dl1,dl2 = l1-l0,l2-l0
        l1,l2 = l0+spec_kfwhm*dl1,l0+spec_kfwhm*dl2     
        ssp = sp.subspec(lmin=l1, lmax=l2)
        
        return ima,ssp 
    
    def truncate_spec(self, ins, spec, kfwhm):
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        temp = ins['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        rspec = spec.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'), 
                            shape=temp.shape[0])
        rspec.data *= spec.data.sum()/rspec.data.sum()        
        nl1 = l0 - kfwhm*(l0-l1)
        nl2 = l0 + kfwhm*(l2-l0)
        if (nl1 < rspec.get_start()) or (nl2 > rspec.get_end()):
            raise ValueError(f'adaptive spectra truncation outside spec limits')
        tspec = rspec.subspec(lmin=nl1, lmax=nl2)
        return tspec
        
    def adaptative_circular_aperture(self, ins, ima, kfwhm):
        peak = ima.peak()
        center = (peak['p'],peak['q'])
        fwhm,_ = ima.fwhm(center=center, unit_center=None, unit_radius=None)
        rad = kfwhm*fwhm
        tima = self.fixed_circular_aperture(ins, ima, rad)
        return tima,fwhm,rad
    
    def fixed_circular_aperture(self, ins, ima, radius):
        peak = ima.peak()
        rad = radius*ima.oversamp/ins['spaxel_size']
        center = (peak['p'],peak['q'])
        tima = ima.copy()
        x, y = np.meshgrid((np.arange(ima.shape[1]) - center[1]),
                               (np.arange(ima.shape[0]) - center[0]))
        ksel = (x**2 + y**2 > rad**2)    
        tima.data[ksel] = 0
        rima = tima.rebin(ima.oversamp)
        rima.data *= tima.data.sum()/rima.data.sum()
        ksel = rima.data == 0
        rima.mask_selection(ksel)
        rima.crop()
        return rima    

    def snr_from_source(self, ins, flux, ima, spec, moon, debug=True):
        """ compute snr cube and on an aperture for a source defined
            by flux x image x spectra
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            flux in erg/s/cm2
            moon condition [darksky, greysky]
            obs parameters: kfwhm_spec, kfwhm_ima
        """
        self._check_moon(moon)
        obs = self.obs
        # adaptive truncation of spectrum
        tspec = self.truncate_spec(ins, spec, obs['kfwhm_spec'])
        # truncation of image
        if ins['type'] == 'IFS':
            # adaptive truncation of image
            tima,fwhm,rad = self.adaptative_circular_aperture(ins, ima, obs['kfwhm_ima'])
            if debug:
                self.logger.debug('Adaptive circular aperture FWHM %.2f Radius %.2f',fwhm,rad)
        elif ins['type'] == 'MOS':
            tima = self.fixed_circular_aperture(ins, ima, 0.5*ins['aperture'])
        dl = ins['dlbda']
        cube = flux * tima * tspec
        res = self.snr_from_cube(ins, cube, moon)
        # sum over an aperture spatial and spectral
        frac_ima = np.sum(tima.data)
        frac_spec = np.sum(tspec.data)
        nspaxels = np.count_nonzero(tima.data)
        nspectels = tspec.data.shape[0]
        nvoxels = nspaxels * nspectels
        # compute total snr and noise      
        vartot = res['noise']['tot'].data**2
        nph_source = np.sum(res['nph_source'].data)
        nph_sky=np.sum(res['nph_sky'].data)*nspaxels
        snr = nph_source/np.sqrt(np.sum(vartot))
        # fraction of flux recovered
        frac_flux = frac_ima*frac_spec
        # save data for the aperture
        res['aper'] = dict(frac_flux=frac_flux,
                           frac_ima=frac_ima,
                           frac_spec=frac_spec,
                           nspaxels=nspaxels,
                           nspectels=nspectels,
                           nvoxels=nvoxels,
                           snr=snr,
                           nph_source=nph_source,
                           nph_sky=nph_sky,
                           ron=np.sqrt(nvoxels)*res['noise']['ron'],
                           dark=np.sqrt(nvoxels)*res['noise']['dark'],
                           sky_noise=np.sqrt(nph_sky),
                           source_noise=np.sqrt(nph_source),
                           tot_noise=np.sqrt(np.sum(vartot)),
                           )
        if debug:
            self.logger.debug('SN %.1f FracFlux %.2f Nvoxels %d',snr,frac_flux,nvoxels)
        res['dl'] = dl
        res['flux'] = flux
        res['moon'] = moon
        res['trunc_ima'] = tima
        res['trunc_spec'] = tspec       
        return res   
  
    def snr_from_cube(self, ins, cube, moon):
        obs = self.obs
        # truncate instrans and sky to the wavelength limit of the input cube
        ins_sky = ins[moon].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_atm = ins['atmtrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = cube.wave.get_step(unit='Angstrom')
        w = cube.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins_atm.data**obs['airmass'])
        Kt =  ins_ins * a
        nph_source = cube.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt[:,np.newaxis,np.newaxis].data * cube.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = np.sqrt(nph_sky)
        source_noise = cube.copy()
        source_noise.data = np.sqrt(nph_source.data)
        skynoise_cube = np.tile(sky_noise.data[:,np.newaxis,np.newaxis], (1,cube.shape[1],cube.shape[2]))
        tot_noise = cube.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + skynoise_cube.data**2 + source_noise.data**2)
        ksel = cube.data == 0
        tot_noise.data[ksel] = 0
        snr = cube.copy()
        snr.data = nph_source.data / tot_noise.data 
        res = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr_cube=snr, flux_source=cube, nph_source=nph_source, nph_sky=nph_sky)
        return res 
    
    def _snr_from_spec(self, spec, moon, channel):
        """ spec input spectrum in erg/s/cm2/spectel
            return snr spectrum """
        WIP
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
            
    def flux_from_source(self, ins, snr, ima, spec, moon, bracket=(0.1,1000)):
        """ compute flux for a given snr and source defined
            by flux x image x spectra
            flux in erg/s/cm2
            ins instrument (eg self.ifs['blue'] or self.moslr['red'])
            moon condition [darksky, greysky]
        """ 
        res0 = root_scalar(self.fun, args=(snr, ins, ima, spec, moon), method='brentq', 
                              bracket=bracket, rtol=1.e-3, maxiter=100)
        flux = res0.root*1.e-20
        res = self.snr_from_source(ins, flux, ima, spec, moon)
        res['aper']['flux'] = flux
        return res
    
    def fun(self, flux, snr0, ins, ima, spec, moon):
        res = self.snr_from_source(ins, flux*1.e-20, ima, spec, moon, debug=False)
        snr = res['aper']['snr']
        #print(flux,snr)
        return snr-snr0  
        
        
        
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

def peakwave_asymgauss(lpeak, sigma, skew, dl=0.01):
    """ return l0 so that the asymgauss(l0) peaked at l0
    """
    wave = np.arange(lpeak-5*sigma,lpeak+5*sigma,dl)
    res0 = root_scalar(_funasym, args=(lpeak, sigma, skew, wave), method='brentq',
                          bracket=[lpeak-2*sigma,lpeak+2*sigma], rtol=1.e-3, maxiter=100) 
    return res0.root

def _funasym(l0, lpeak, sigma, skew, wave):
    f = asymgauss(1.0, l0, sigma, skew, wave)
    k = f.argmax()
    zero = wave[k] - lpeak
    #print(zero)
    return zero

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

def moffat(samp, fwhm, beta, kfwhm=2, oversamp=1):
    ns = (int((kfwhm*fwhm/samp+1)/2)*2 + 1)*oversamp
    pixfwhm = oversamp*fwhm/samp
    ima = moffat_image(fwhm=(pixfwhm,pixfwhm), n=beta, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.data /= ima.data.sum()
    return ima

def compute_sky(lbda1, lbda2, dlbda, lsf, moon):
    """ return sky table model
    lbda1,lbda2 range in wavelength A
    dlbda step in wavelength A
    lsf convolution in pixels
    moon (darksky, greysky)
    """
    skyModel = SkyModel()
    if moon == 'darksky':
        sep = 0
    elif moon == 'greysky':
        sep = 90
    else:
        raise ValueError(f'Error in moon {moon}')
    skypar = dict(wmin=lbda1*0.1, wmax=lbda2*0.1, wdelta=dlbda*0.1, 
                  lsf_gauss_fwhm=lsf, lsf_type='Gaussian',
                  moon_sun_sep=sep, observatory='paranal')
    skypar = fixObservatory(skypar)
    skyModel.callwith(skypar)
    f = BytesIO()
    f = BytesIO(skyModel.data)
    tab = Table.read(f)
    return tab

def show_noise(res, ax, legend=False):
    r = res['noise']
    rtot = r['tot']
    f = (r['sky'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='r', label='sky' if legend else None)
    f = (r['source'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='b', label='source' if legend else None)
    f = (r['ron']/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='g', label='ron' if legend else None)
    f = (r['dark']/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='m', label='dark' if legend else None)
    f = (r['ron']**2 + r['dark']**2)/rtot.data**2
    ax.plot(rtot.wave.coord(), f, color='k', label='ron+dark' if legend else None)
    if legend:
        ax.legend(loc='upper right')
        ax.axhline(0.5, color='k', alpha=0.2)
        


def get_snr_ima_spec(res):
    var = res['noise']['tot'].copy()
    var.data = var.data**2
    noise_ima = var.sum(axis=0)
    noise_ima.data = np.sqrt(noise_ima.data)
    source_ima = res['nph_source'].sum(axis=0)
    snr_ima = source_ima/noise_ima
    noise_sp = var.sum(axis=(1,2))
    noise_sp.data = np.sqrt(noise_sp.data)
    source_sp = res['nph_source'].sum(axis=(1,2))
    snr_sp = source_sp/noise_sp
    return snr_ima,snr_sp