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
from scipy.optimize import root_scalar, minimize_scalar
from skycalc_cli.skycalc import SkyModel
from skycalc_cli.skycalc_cli import fixObservatory
from io import BytesIO
import glob
#from collections import OrderedDict
from spextra import Spextrum

# global variables
g_frac_ima = None
g_size_ima = None
g_nspaxels = None



class ETC:
    """ Generic class for Exposure Time Computation (ETC) """ 
   
    def __init__(self, log=logging.INFO):
        self.version = __version__
        setup_logging(__name__, level=log, stream=sys.stdout) 
        self.logger = logging.getLogger(__name__)
        
    def set_logging(self, log):
        """ Change logging value

        Parameters
        ----------
        log : str
             desired log mode "DEBUG","INFO","WARNING","ERROR"
            
        """
        
        self.logger.setLevel(log)
                               
    
    def _info(self, ins_names):
        """ print detailed information

        Parameters
        ----------
        ins_names : list of str
               list of instrument names (e.g ['ifs','moslr'])

        """        
        self.logger.info('%s ETC version: %s', self.name, self.version)
        self.logger.info('Diameter: %.2f m Area: %.1f m2', self.tel['diameter'],self.tel['area'])
        for ins_name in ins_names: 
            insfam = getattr(self, ins_name)
            for chan in insfam['channels']:
                ins = insfam[chan]
                self.logger.info('%s type %s Channel %s', ins_name.upper(), ins['type'], chan)                
                self.logger.info('\t %s', ins['desc'])
                self.logger.info('\t Spaxel size: %.2f arcsec Image Quality tel+ins fwhm: %.2f arcsec beta: %.2f ', ins['spaxel_size'], ins['iq_fwhm'], ins['iq_beta'])
                if 'aperture' in ins.keys():
                    self.logger.info('\t Fiber aperture: %.1f arcsec', ins['aperture'])
                self.logger.info('\t Wavelength range %s A step %.2f A LSF %.1f pix Npix %d', ins['instrans'].get_range(), 
                                  ins['instrans'].get_step(), ins['lsfpix'], ins['instrans'].shape[0])
                self.logger.info('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                                  ins['instrans'].data.max(), ins['instrans'].wave.coord(ins['instrans'].data.argmax()),
                                  ins['instrans'].data.min(), ins['instrans'].wave.coord(ins['instrans'].data.argmin()))
                self.logger.info('\t Detector RON %.1f e- Dark %.1f e-/h', ins['ron'],ins['dcurrent'])
                for sky in ins['sky']:
                    self.logger.info('\t Sky moon %s airmass %s table %s', sky['moon'], sky['airmass'], 
                                      os.path.basename(sky['filename']))
                self.logger.info('\t Instrument transmission table %s', os.path.basename(ins['instrans'].filename))          
        
              
    def set_obs(self, obs):
        """save obs dictionary to self

        Parameters
        ----------
        obs : dict 
            dictionary of observation parameters

        """
        if ('ndit' in obs.keys()) and ('dit' in obs.keys()):
            obs['totexp'] = obs['ndit']*obs['dit']/3600.0 # tot integ time in hours
        self.obs = obs
        
    def get_spectral_resolution(self, ins):
        """ return spectral resolving power

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])

        Returns
        -------
        numpy array
            spectral resoving power (lbda/dlbda)

        """
        lsf = ins['lsfpix']*ins['dlbda']
        wave = ins['wave'].coord()
        res = wave/lsf
        return res

    def get_sky(self, ins, moon):
        """ return sky emission and tranmission spectra

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        moon : str
            moon observing condition (e.g "darksky")

        Returns
        -------
        tuple of MPDAF spectra
            emission and absorption sky spectra

        """
        airmass = self.obs['airmass']
        for sky in ins['sky']:
            if np.isclose(sky['airmass'], airmass) and (sky['moon'] == moon):
                return sky['emi'],sky['abs']
        raise ValueError(f"moon {moon} airmass {airmass} not found in loaded sky configurations")
    
    def get_spec(self, ins, dspec, oversamp=10, lsfconv=True):
        
        """ compute source spectrum from the model parameters
        
        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        dspec : dict
            dictionary of parameters describing the source spectrum
        oversamp : int
             oversampling factor (Default value = 10)
        lsfconv : bool
             apply LSF convolution (Default value = True)
        
        Returns
        -------
        MPDAF spectrum
            resulting source spectrum
        
        """
        lstep = ins['instrans'].get_step()
        l1,l2 = ins['instrans'].get_start(),ins['instrans'].get_end()
        if dspec['type'] == "flatcont":
            dlbda = ins['dlbda']
            wave = ins['instrans'].wave
            k = wave.pixel(dspec['wave'][0], nearest=True)
            l1 = wave.coord(k)
            k = wave.pixel(dspec['wave'][1], nearest=True)
            l2 = wave.coord(k)            
            npts = np.int((l2 - l1)/dlbda + 1.5)
            spec = Spectrum(wave=WaveCoord(cdelt=dlbda, crval=l1), data=np.ones(npts))
            oversamp = 1  # we do not oversamp for flatcont
        elif dspec['type'] == "template":
            # get template spectrum
            name = dspec['name']
            sp = Spextrum(name)
            l0,dl = dspec['wave_center'],dspec['wave_width']
            w,y = sp._get_arrays(wavelengths=None)
            w,y = w.value,y.value
            dw = w[1] - w[0]
            spec0 = Spectrum(data=y, wave=WaveCoord(cdelt=dw, crval=w[0])) 
            # normalise to 1 in the given window
            if ((l0-dl/2) < w[0]) or ((l0+dl/2) > w[-1]):
                raise ValueError('wave range outside template wavelength limits')
            vmean = spec0.mean(lmin=l0-dl/2,lmax=l0+dl/2)[0]
            spec0.data /= vmean                        
            # resample 
            rspec = spec0.resample(lstep, start=l1)
            rspec = rspec.subspec(lmin=l1, lmax=l2)
            # LSF convolution
            if lsfconv:
                spec = rspec.filter(width=ins['lsfpix'])
            else:
                spec = rspec
            oversamp = 1  # we do not oversamp for template
        elif dspec['type'] == 'line':
            kfwhm = dspec.get('kfwhm', 5)
            dl = kfwhm*10*dspec['sigma']
            if dspec['skew'] == 0:
                l0 = dspec['lbda']
            else:
                l0 = peakwave_asymgauss(dspec['lbda'], dspec['sigma'], dspec['skew'])
            wave = np.arange(dspec['lbda']-dl,dspec['lbda']+dl,lstep/oversamp)
            f = asymgauss(1.0, l0, dspec['sigma'], dspec['skew'], wave)
            sp = Spectrum(data=f, wave=WaveCoord(cdelt=(wave[1]-wave[0]), crval=wave[0]))
            l0,l1,l2 = fwhm_asymgauss(sp.wave.coord(),sp.data)
            dl1,dl2 = l1-l0,l2-l0
            l1,l2 = l0+kfwhm*dl1,l0+kfwhm*dl2     
            rspec = sp.subspec(lmin=l1, lmax=l2)
            # LSF convolution 
            if lsfconv:
                spec = rspec.filter(width=ins['lsfpix']*oversamp)
            else:
                spec = rspec
        else:
            raise ValueError('Unknown spectral type')                  
        spec.oversamp = oversamp
        return spec
    
    def get_ima(self, ins, dima, oversamp=10, uneven=1):
        """ compute source image from the model parameters
        
         Parameters
         ----------
         ins : dict
             instrument (eg self.ifs['blue'] or self.moslr['red'])
         dima : dict
             dictionary of parameters describing the source spectrum
         oversamp : int
             oversampling factor (Default value = 10)
         uneven : int
              if 1 the size of the image will be uneven (Default value = 1)
 
         Returns
         -------
         MPDAF image
             image of the source

         """
        if dima['type'] == 'moffat':
            kfwhm = dima.get('kfwhm', 3)
            ima = moffat(ins['spaxel_size'], dima['fwhm'], dima['beta'], dima.get('ell',0),
                         kfwhm=kfwhm, oversamp=oversamp, uneven=uneven)
        ima.oversamp = oversamp 
        return ima


    def truncate_spec_adaptative(self, ins, spec, kfwhm):
        """ truncate an emission line spectrum as function of the line FWHM
            the window size is compute as center +/- kfwhm*fwhm 


        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPFAF spectrum
            source spectrum 
        kfwhm : float
            factor relative to the line FWHM, 
        Returns
        -------
        tuple
             tspec truncated MPDAF spectrum 
             waves numpy array of corresponding wavelengths (A)
             nspectels (int) number of spectels kept
             size (float) wavelength range (A)
             frac_flux (float) fraction flux kept after truncation 

        """
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        temp = ins['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        rspec = spec.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'), 
                            shape=temp.shape[0])
        # WIP check if rspec has a shape > 0
        rspec.data *= spec.data.sum()/rspec.data.sum()        
        nl1 = l0 - kfwhm*(l0-l1)
        nl2 = l0 + kfwhm*(l2-l0)
        if (nl1 < rspec.get_start()) or (nl2 > rspec.get_end()):
            raise ValueError(f'adaptive spectra truncation outside spec limits')
        if nl2-nl1 <= rspec.get_step():
            raise ValueError(f'kfwhm is too small {kfwhm}')
        tspec = rspec.subspec(lmin=nl1, lmax=nl2)        
        waves = tspec.wave.coord()
        nspectels = tspec.shape[0]
        size = tspec.get_end() - tspec.get_start()
        frac_flux = tspec.data.sum()
        return tspec,waves,nspectels,size,frac_flux 
    
    def optimum_spectral_range(self, ins, flux, ima, spec):
        """ compute the optimum window range which maximize the S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum

        Returns
        -------
        float
            factor relative to FWHM (kfwhm)

            the kfwhm value is also updated into the obs dictionary
        """        
        obs = self.obs
        if obs['spec_range_type'] != 'adaptative':
            raise ValueError('obs spec_range_type must be set to adaptative')        
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        kmax = min((spec.get_end()-l0)/(l2-l0),(l0-spec.get_start())/(l0-l1))
        kmin = max(spec.get_step()/(l2-l0), spec.get_step()/(l0-l1))
        bracket = [5*kmin,0.9*kmax]
        self.logger.debug('Optimizing kwhm in %s', bracket)
        is_ps = False
        if (ima is None) and (obs['ima_type'] == 'ps'):
            is_ps = True
            # we compute the PSF at the central wavelength
            ima = self.get_image_psf(ins, l0)
            obs['ima_type'] = 'resolved'
        res = minimize_scalar(_fun_range, args=(self, ins, flux, ima, spec), 
                              bounds=bracket, options=dict(xatol=0.01), method='bounded')
        kfwhm = res.x
        snr = -res.fun
        self.obs['spec_range_kfwhm'] = kfwhm
        tspec,waves,nspectels,size,frac = self.truncate_spec_adaptative(ins, spec, kfwhm)
        if is_ps:
            obs['ima_type'] = 'ps' # set ima_type back to ps
        self.logger.debug('Optimum spectral range nit=%d kfwhm=%.2f S/N=%.1f Size=%.1f Flux=%.2e Frac=%.2f',res.nfev,kfwhm,snr,size,flux,frac)
        return res.x   
    
    
    def truncate_spec_fixed(self, ins, spec, nsp):
        """ truncate the spectrum to a fixed spectral window size

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            source spectrum
        nsp : int
            half number of spectels to use (size is 2 * nsp + 1)

        Returns
        -------
        tuple
             tspec truncated MPDAF spectrum 
             waves numpy array of corresponding wavelengths (A)
             nspectels (int) number of spectels kept
             size (float) wavelength range (A)
             frac_flux (float) fraction flux kept after truncation 

        """        
        l0,l1,l2 = fwhm_asymgauss(spec.wave.coord(), spec.data)
        temp = ins['instrans'].subspec(lmin=spec.get_start(), lmax=spec.get_end())
        rspec = spec.resample(temp.get_step(unit='Angstrom'), start=temp.get_start(unit='Angstrom'), 
                            shape=temp.shape[0])
        rspec.data *= spec.data.sum()/rspec.data.sum() 
        k0 = rspec.wave.pixel(l0, nearest=True)
        tspec = rspec[k0-nsp:k0+nsp+1]
        nspectels = 2*nsp+1
        size = nspectels*ins['dlbda']
        frac_flux = tspec.data.sum()
        if nsp == 0:
            tspec = tspec.data[0]
            waves = rspec.wave.coord(k0)
        else:
            waves = tspec.wave.coord()
        return tspec,waves,nspectels,size,frac_flux   
        
    def adaptative_circular_aperture(self, ins, ima, kfwhm):
        """ truncate the image with a window  size relative to the image FWHM

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        kfwhm : float
            factor relative to FWHM to define the truncation aperture

        Returns
        -------
        tuple
             tima truncated MPDAF image 
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation+
        """        
        peak = ima.peak()
        center = (peak['p'],peak['q'])
        fwhm,_ = ima.fwhm_gauss(center=center, unit_center=None, unit_radius=None)
        rad = kfwhm*fwhm*ins['spaxel_size']/ima.oversamp
        tima,nspaxels,size_ima,frac_ima = self.fixed_circular_aperture(ins, ima, rad)
        return tima,nspaxels,size_ima,frac_ima
    
    def optimum_circular_aperture(self, ins, flux, ima, spec, bracket=[1,5], lrange=None):
        """ compute the optimum aperture which maximize the S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum
        bracket : tuple
             (Default value = [1,5]) :
            
        lrange : tuple
             wavelength range to compute S/N for cont spectrum (Default value = None)

        Returns
        -------
        float
            factor relative to FWHM (kfwhm)

            the kfwhm value is also updated into the obs dictionary

        """
        obs = self.obs
        if obs['ima_aperture_type'] != 'circular_adaptative':
            raise ValueError('obs ima_aperture_type must be set to circular_adaptative')
        if obs['spec_type'] == 'cont':
            if lrange is None:
                raise ValueError('lrange must be set when spec_type is cont')
            krange = spec.wave.pixel(lrange, nearest=True)
        else:
            krange = None 
        is_ps = False
        if (ima is None) and (obs['ima_type'] == 'ps'):
            is_ps = True
            # we compute the PSF at the central wavelength
            l0 = 0.5*(spec.get_end() + spec.get_start())
            ima = self.get_image_psf(ins, l0)
            obs['ima_type'] = 'resolved'
        res = minimize_scalar(_fun_aper, args=(self, ins, flux, ima, spec, krange), 
                              bracket=bracket, tol=0.01, method='brent')
        kfwhm = res.x
        snr = -res.fun
        self.obs['ima_kfwhm'] = kfwhm
        tima,nspaxels,size_ima,frac_ima = self.adaptative_circular_aperture(ins, ima, kfwhm)
        self.logger.debug('Optimum circular aperture nit=%d kfwhm=%.2f S/N=%.1f Aper=%.1f Flux=%.2e Frac=%.2f',res.nit,kfwhm,snr,size_ima,flux,frac_ima)
        if is_ps:
            obs['ima_type'] = 'ps' # set ima_type back to ps
        return res.x   
    
    def fixed_circular_aperture(self, ins, ima, radius, cut=1/16):
        """ truncate the image with a fixed size aperture

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        radius : float
            aperture radius in arcsec
        cut : float
            spaxels with flux below cut will not be counted (Default value = 1/16)

        Returns
        -------
        tuple
             tima truncated MPDAF image 
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation

        """        
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
        ksel = rima.data < cut*rima.data.max()
        rima.mask_selection(ksel)
        rima.crop()
        size = 2*radius
        frac_flux = rima.data.sum()
        ksel = rima.data <= cut*rima.data.max()
        rima.data[ksel] = 0
        rima.mask_selection(ksel)
        nspaxels = np.count_nonzero(rima.data)
        return rima,nspaxels,size,frac_flux 
    
    def square_aperture(self, ins, ima, nsp): 
        """ truncate an image on a squared aperture

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image
        nsp : int
            the number of spaxels to use is 2 * nsp +1

        Returns
        -------
        tuple
             tima truncated MPDAF image 
             nspaxels (int) number of spaxels
             size (float) aperture size (diameter, arcsec)
             frac_flux (float) fraction flux kept after truncation

        """        
        rima = ima.rebin(ima.oversamp)
        rima.data *= ima.data.sum()/rima.data.sum()
        nspaxels = (2*nsp+1)**2
        peak = rima.peak()
        p,q = int(peak['p']+0.5),int(peak['q']+0.5)
        tima = rima[p-nsp:p+nsp+1, q-nsp:q+nsp+1]
        frac_flux = np.sum(tima.data)
        size = (2*nsp+1)*ins['spaxel_size'] 
        return tima,nspaxels,size,frac_flux      
        
    def get_psf_frac_ima(self, ins, flux, spec, lrange=None, oversamp=10, lbin=1):
        """ compute the flux fraction evolution with seeing for a point source

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image
        spec : MPDAF spectrum
            source spectrum           
        lrange : tuple
             wavelength range to compute S/N for cont spectrum (Default value = None)
        oversamp : int
             oversampling factor (Default value = 10)
        lbin : int
             step in wavelength (Default value = 1)

        Returns
        -------
        tuple
            frac_ima (MPDAF spectrum) fraction of flux as function of wavelength
            size_ima (MPDAF spectrum) diameter in arcsec of the aperture
            nspaxels (numpy array of int) corresponding number of spaxels within the aperture

         """
        obs = self.obs
        moon = obs['moon']
        if obs['ima_type'] != 'ps':
            raise ValueError('get_psf_frac_ima only work in ps ima_type')
        obs['ima_type'] = 'resolved' # switch to resolved for the computation
        fwhm = self.get_image_quality(ins)
        beta = ins['iq_beta']
        # if IFS and adaptative, compute the optimal parameters for the two extreme wavelength
        if (ins['type'] == 'IFS') and (obs['ima_aperture_type'] == 'circular_adaptative'):
            self.logger.debug('Computing optimum values for kfwhm')
            if lrange is None:
                l0 = 0.5*(spec.get_end() + spec.get_start())
                dl = spec.get_end() - spec.get_start()
                lrange = [l0 - 0.4*dl, l0 + 0.4*dl]
            kfwhm_edges = []
            for k in [0,-1]:
                ima = moffat(ins['spaxel_size'], fwhm.data[k], beta, oversamp=oversamp)
                ima.oversamp = oversamp
                kfwhm_edges.append(self.optimum_circular_aperture(ins, flux, ima, spec, lrange=lrange))
            self.logger.debug('Optimum values of kfwhm at wavelengths edges: %s', kfwhm_edges)
        
        # loop on wavelength
        frac_ima = spec.copy()
        size_ima = spec.copy()
        waves = spec.wave.coord()
        nspaxels = np.zeros(len(waves), dtype=int)
        if lbin > 1:
            klist = np.linspace(0, len(waves)-1, lbin, dtype=int)
        else:
            klist = range(len(waves))
        self.logger.debug('Computing frac and nspaxels for %d wavelengths (lbin %d)', len(klist), lbin)
        for k in klist:
            wave = waves[k]
            ima = moffat(ins['spaxel_size'], fwhm.data[k], beta, oversamp=oversamp)
            ima.oversamp = oversamp
            if ins['type'] == 'IFS':
                if obs['ima_aperture_type'] == 'square_fixed':
                    tima,nspa,size,frac = self.square_aperture(ins, ima, obs['ima_aperture_hsize_spaxels'])
                elif obs['ima_aperture_type'] == 'circular_adaptative':   
                    kfwhm = np.interp(wave, [waves[0],waves[-1]], kfwhm_edges)
                    tima,nspa,size,frac = self.adaptative_circular_aperture(ins, ima, kfwhm)
            elif ins['type'] == 'MOS':
                tima,nspa,size,frac  = self.fixed_circular_aperture(ins, ima, 0.5*ins['aperture'])
            frac_ima.data[k] = frac
            size_ima.data[k] = size
            nspaxels[k] = nspa
        if lbin > 1:
            self.logger.debug('Performing interpolation')
            ksel = nspaxels == 0
            for k in np.arange(0, len(waves))[ksel]:
                frac_ima.data[k] = np.interp(waves[k],waves[~ksel],frac_ima.data[~ksel])
                size_ima.data[k] = np.interp(waves[k],waves[~ksel],size_ima.data[~ksel])
                nspaxels[k] = int(np.interp(waves[k],waves[~ksel],nspaxels[~ksel]) + 0.5)
            
        for k in [0,-1]:
            self.logger.debug('At %.1f A  FWHM: %.2f Flux fraction: %.2f Aperture: %.1f Nspaxels: %d',waves[k],fwhm.data[k],frac_ima.data[k],
                          size_ima.data[k], nspaxels[k])
        obs['ima_type'] = 'ps' # switch back to ps 
            
        return frac_ima,size_ima,np.array(nspaxels)
  

    def snr_from_source(self, ins, flux, ima, spec, loop=False, debug=True):
        """ main routine to perform the S/N computation for a given source

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum          
        loop : bool
             set to True for multiple call (used only in ps and cont, Default value = False)
        debug :
             if True print some info in logger.debug mode (Default value = True)

        Returns
        -------
        dict
            result dictionary (see documentation)

         """        
        
        obs = self.obs
        usedobs = {}
        _checkobs(obs, usedobs, ['moon','dit','ndit','airmass','spec_type','ima_type'])
        tflux = flux
        is_cont = False
        is_line = False
        if obs['spec_type'] == 'cont': # flux is in erg/s/cm2/A
            is_cont = True
            # convert flux if type is cont
            tspec = spec.copy()
            tflux *= ins['dlbda'] # flux in erg/s/cm2/spectel
            nspectels = 1
            frac_spec = 1
            size_spec = spec.get_end() - spec.get_start()
            waves = tspec.wave.coord()
        elif obs['spec_type'] == 'line': #flux in erg/s/cm2
            _checkobs(obs, usedobs, ['spec_range_type'])
            is_line = True
            # truncate spectrum if type is line
            if obs['spec_range_type'] == 'fixed':
                _checkobs(obs, usedobs, ['spec_range_hsize_spectels'])
                tspec,waves,nspectels,size_spec,frac_spec = self.truncate_spec_fixed(ins, spec, obs['spec_range_hsize_spectels'])
            elif obs['spec_range_type'] == 'adaptative': 
                # adaptive truncation of spectrum
                _checkobs(obs, usedobs, ['spec_range_kfwhm'])
                tspec,waves,nspectels,size_spec,frac_spec = self.truncate_spec_adaptative(ins, spec, obs['spec_range_kfwhm']) 
            else:
                raise ValueError('Unknown spec_range_type')
                
                       
        is_ps = False 
        is_sb = False
        if (ima is None) :
            if (obs['ima_type'] != 'ps') and (obs['ima_type'] != 'sb'):
                raise ValueError('ima can be None only for ps or sb obs ima_type')
            
            if (obs['ima_type'] == 'ps'):
                # use the seeing PSF to compute frac_ima and nspaxels evolution with wavelength
                if is_cont:
                    if loop:
                        frac_ima,size_ima,nspaxels = g_frac_ima,g_size_ima,g_nspaxels
                    else:
                        lbin = 20 if spec.shape[0]>100 else 1
                        frac_ima,size_ima,nspaxels = self.get_psf_frac_ima(ins, flux, spec, lbin=lbin)
                    frac_flux = frac_ima.copy()
                    frac_flux.data = frac_ima.data*frac_spec
                if is_line:
                    _checkobs(obs, usedobs, ['seeing'])
                    # use a constant PSF computed at the central wavelength the emission line
                    l0 = 0.5*(tspec.get_end() + tspec.get_start())
                    fwhm = get_seeing_fwhm(obs['seeing'], obs['airmass'], l0, self.tel['diameter'], ins['iq_fwhm'])
                    ima = moffat(ins['spaxel_size'], fwhm, ins['iq_beta'], oversamp=10) 
                    ima.oversamp = 10
                    if debug:
                        self.logger.debug('Computing PSF at %.1f fwhm %.2f beta %.1f',l0,fwhm,ins['iq_beta'])
                    obs['ima_type'] = 'resolved' # change temporarily ima_type
                is_ps = True
             
            if (obs['ima_type'] == 'sb') :     
                # surface brightness
                frac_ima = 1        
                tflux *= ins['spaxel_size']**2 #erg.s-1.cm-2/voxels
                if ins['type'] == 'IFS':
                    _checkobs(obs, usedobs, ['ima_area'])
                    nspaxels = int(obs['ima_area']/ins['spaxel_size']**2 + 0.5)
                    size_ima = np.sqrt(obs['ima_area']) # assuming square area
                    area_aper = obs['ima_area']
                elif ins['type'] == 'MOS':
                    nspaxels = int((2*np.pi*ins['aperture']**2/4)/ins['spaxel_size']**2 + 0.5)
                    size_ima = ins['aperture']
                    area_aper = np.pi*size_ima**2/4
                frac_flux = frac_ima*frac_spec
                
                is_sb = True
            
        if obs['ima_type'] == 'resolved':
            # truncate image if type is resolved
            if ima is None:
                raise ValueError('image cannot be none for resolved image type')
            if ins['type'] == 'IFS':
                _checkobs(obs, usedobs, ['ima_aperture_type'])
                if obs['ima_aperture_type'] == 'square_fixed':
                    _checkobs(obs, usedobs, ['ima_aperture_hsize_spaxels'])
                    tima,nspaxels,size_ima,frac_ima = self.square_aperture(ins, ima, obs['ima_aperture_hsize_spaxels'])
                    area_aper = size_ima**2
                elif obs['ima_aperture_type'] == 'circular_adaptative': 
                    _checkobs(obs, usedobs, ['ima_kfwhm'])
                    # adaptive truncation of image
                    tima,nspaxels,size_ima,frac_ima = self.adaptative_circular_aperture(ins, ima, obs['ima_kfwhm'])
                    area_aper = np.pi*size_ima**2/4
                    if debug:
                        self.logger.debug('Adaptive circular aperture diameter %.2f frac_flux %.2f',size_ima,frac_ima)
                else:
                    raise ValueError(f"unknown ima_aperture_type {obs['ima_aperture_type']}")
            elif ins['type'] == 'MOS':
                size_ima = ins['aperture']
                tima,nspaxels,size_ima,frac_ima  = self.fixed_circular_aperture(ins, ima, 0.5*ins['aperture'])
                area_aper = np.pi*size_ima**2/4
            frac_flux = frac_ima*frac_spec                 
        
        # perform snr computation
        if isinstance(tspec, (float)):
            ima_data = tflux * tima * tspec
            ima_data.data.mask = tima.data.mask
            res = self.snr_from_ima(ins, ima_data, waves)
        elif is_sb:
            spec_data = tflux * tspec
            res = self.snr_from_spec(ins, spec_data)
        elif is_ps and is_cont:
            spec_data = tflux * tspec
            res = self.snr_from_ps_spec(ins, spec_data, frac_ima, nspaxels)  
            res['spec']['frac_spec'] = frac_spec
        else:
            cube_data = tflux * tima * tspec
            res = self.snr_from_cube(ins, cube_data)
            
        # compute additionnal results
        if (not is_ps) or (obs['ima_type'] == 'resolved'): 
            resc = res['cube']
            nvoxels = nspaxels*nspectels
            dl = ins['dlbda']            
            vartot = resc['noise']['tot'].copy()
            vartot.data = vartot.data**2        
            # sum over spatial axis to get spectra values res['spec']
            nph_sky = resc['nph_sky']*nspaxels
            tot_noise = nph_sky.copy()        
            if obs['ima_type'] == 'resolved':
                nph_source = resc['nph_source'].sum(axis=(1,2))
                tot_noise.data = np.sqrt(vartot.sum(axis=(1,2)).data)
            elif obs['ima_type'] == 'sb':
                nph_source = resc['nph_source']*nspaxels
                tot_noise.data = np.sqrt(vartot.data*nspaxels)        
            snr = nph_sky.copy()
            snr.data = nph_source.data / tot_noise.data
            sky_noise = nph_sky.copy()
            sky_noise.data = np.sqrt(nph_sky.data)
            source_noise = nph_source.copy()
            source_noise.data = np.sqrt(nph_source.data) 
            res['spec'] = dict(snr=snr,
                               snr_mean=snr.data.mean(),
                               snr_max=snr.data.max(),
                               snr_min=snr.data.min(),
                               frac_flux=frac_flux,
                               frac_ima=frac_ima,
                               frac_spec=frac_spec,
                               nb_spaxels=nspaxels,
                               nph_source=nph_source,
                               nph_sky=nph_sky,
                               noise = dict(ron=np.sqrt(nspaxels)*resc['noise']['ron'],
                                            dark=np.sqrt(nspaxels)*resc['noise']['dark'],
                                            sky=sky_noise,
                                            source=source_noise,
                                            tot=tot_noise,
                                            )
                           ) 
        # if spec type is line summed over spectral axis to get aperture values res['aper]
        if obs['spec_type'] == 'line':
            sp = res['spec']
            nph_source_aper = sp['nph_source'].data.sum()
            nph_sky_aper = sp['nph_sky'].data.sum()
            tot_noise_aper = np.sqrt(np.sum(sp['noise']['tot'].data**2))
            snr_aper = nph_source_aper/tot_noise_aper
            sky_noise_aper = np.sqrt(np.sum(sp['noise']['sky'].data**2))
            source_noise_aper = np.sqrt(np.sum(sp['noise']['source'].data**2))
            ron_aper = np.sqrt(nvoxels)*resc['noise']['ron']
            dark_aper = np.sqrt(nvoxels)*resc['noise']['dark']
            res['aper'] = dict(snr=snr_aper,
                               size=size_ima,
                               area=area_aper,                               
                               frac_flux=frac_flux,
                               frac_ima=frac_ima,
                               frac_spec=frac_spec,
                               nb_spaxels=nspaxels,
                               nb_spectels=nspectels,
                               nb_voxels=nvoxels,
                               nph_source=nph_source_aper,
                               nph_sky=nph_sky_aper,
                               ron=ron_aper,
                               dark=dark_aper,
                               sky_noise=sky_noise_aper,
                               source_noise=source_noise_aper,
                               tot_noise=tot_noise_aper,
                               frac_detnoise=(ron_aper**2+dark_aper**2)/tot_noise_aper**2,
                               ) 
        if debug:
            self.logger.debug('Source type %s & %s Flux %.2e S/N %.1f FracFlux %.3f Nspaxels %d Nspectels %d',
                              obs['ima_type'], obs['spec_type'], flux,
                              snr_aper if 'aper' in res.keys() else res['spec']['snr_mean'],
                              res['spec']['frac_ima'].mean()[0] if is_ps and is_cont else frac_flux,
                              int(np.mean(res['spec']['nb_spaxels'])+0.5) if is_ps and is_cont else nspaxels,
                              nspectels)
        res['input']['dl'] = ins['dlbda']
        res['input']['flux'] = flux
        _copy_obs(usedobs, res['input'])
        if ima is not None:
            res['cube']['trunc_ima'] = tima
        res['cube']['trunc_spec'] = tspec 
        if is_ps and is_line: # set ima_type back to ps
            obs['ima_type'] = 'ps'
        return res   
  
    def snr_from_cube(self, ins, cube):
        """ compute S/N from a data cube

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        cube : MPDAF cube
            source data cube in flux/voxels

        Returns
        -------
        dict
            result dictionary 

            this routine is called by snr_from_source

        """
        
        obs = self.obs
        moon = obs['moon']
        # truncate instrans and sky to the wavelength limit of the input cube
        sky_emi,sky_abs = self.get_sky(ins, moon)
        ins_sky = sky_emi.subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=cube.wave.get_start(), lmax=cube.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = cube.wave.get_step(unit='Angstrom')
        w = cube.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins_atm.data)
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
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=cube, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)            
        return res 
        
    
    def snr_from_ima(self, ins, ima, wave):
        """ compute S/N from an image

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image in flux/spaxels
        wave : float
            wavelength in A

        Returns
        -------
        dict
            result dictionary 

            this routine is called by snr_from_source
        """
        obs = self.obs
        moon = obs['moon']
        # get instrans and sky tvalue at the given wavelength 
        ins_sky = ins[moon].data[ins[moon].wave.pixel(wave, nearest=True)]
        ins_ins = ins['instrans'].data[ins['instrans'].wave.pixel(wave, nearest=True)]
        ins_atm = ins['atmtrans'].data[ins['atmtrans'].wave.pixel(wave, nearest=True)]
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = ins['instrans'].wave.get_step(unit='Angstrom')
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins_atm)
        Kt =  ins_ins * a
        nph_source = ima.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt * ima.data # number of photons received from the source
        nph_sky = ima.copy()
        nph_sky.data[:,:] = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = ima.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = ima.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2)
        ksel = ima.data == 0
        tot_noise.data[ksel] = 0
        snr = ima.copy()
        snr.data = nph_source.data / tot_noise.data 
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=ima, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)        
        return res     
    
    def snr_from_spec(self, ins, spec):
        """compute S/N from a spectrum in flux/spectel

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            source spectrum in flux/spectel

        Returns
        -------
        dict
            result dictionary 

            this routine is called by snr_from_source
            

        """
        obs = self.obs
        moon = obs['moon']
        # truncate instrans and sky to the wavelength limit of the input cube
        sky_emi,sky_abs = self.get_sky(ins, moon)
        ins_sky = sky_emi.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = spec.wave.get_step(unit='Angstrom')
        w = spec.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins_atm.data)
        Kt =  ins_ins * a
        nph_source = spec.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt.data * spec.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = 1
        ron_noise = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = spec.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = spec.copy()
        tot_noise.data = np.sqrt(ron_noise**2 + dark_noise**2 + sky_noise.data**2 + source_noise.data**2)
        snr = spec.copy()
        snr.data = nph_source.data / tot_noise.data 
        res = {}
        res['cube'] = dict(noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise), 
                   snr=snr, nph_source=nph_source, nph_sky=nph_sky)
        res['input'] = dict(flux_source=spec, atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky)
        return res 
    
    def snr_from_ps_spec(self, ins, spec, frac_ima, nspaxels):
        """compute S/N for a point source define by a spectrum in flux/spectel

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        spec : MPDAF spectrum
            point source spectrul in flux/spectel
        frac_ima : MPDAF spectrum
            flux fraction recovered in the aperture as function of wavelength
        nspaxels : numpy array of int
            corresponding number of spaxels in the aperture

        Returns
        -------
        dict
            result dictionary 
            
            this routine is called by snr_from_source

        """
        obs = self.obs
        moon = obs['moon']
        # truncate instrans and sky to the wavelength limit of the input cube
        sky_emi,sky_abs = self.get_sky(ins, moon)
        ins_sky = sky_emi.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_ins = ins['instrans'].subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        ins_atm = sky_abs.subspec(lmin=spec.wave.get_start(), lmax=spec.wave.get_end())
        spaxel_area = ins['spaxel_size']**2
        area = spaxel_area
        dl = spec.wave.get_step(unit='Angstrom')
        w = spec.wave.coord() # wavelength in A
        a = (w*1.e-8/(H_cgs*C_cgs)) * (self.tel['area']*1.e4) * (ins_atm.data)
        Kt =  ins_ins * a
        nph_source = spec.copy()
        nph_source.data = obs['dit'] * obs['ndit'] * Kt.data * spec.data * frac_ima.data # number of photons received from the source
        nph_sky = ins_sky * ins_ins * obs['dit'] * obs['ndit'] * self.tel['area'] * area * (dl/1e4) # nb of photons received from the sky
        nb_voxels = nspaxels
        ron_noise = spec.copy()
        ron_noise.data = np.sqrt(ins['ron']**2*nb_voxels*obs['ndit'])
        dark_noise = spec.copy()
        dark_noise.data = np.sqrt(ins['dcurrent']*nb_voxels*obs['ndit']*obs['dit']/3600)
        sky_noise = nph_sky.copy()
        sky_noise.data = np.sqrt(nph_sky.data)
        source_noise = spec.copy()
        source_noise.data = np.sqrt(nph_source.data)
        tot_noise = spec.copy()
        tot_noise.data = np.sqrt(ron_noise.data**2 + dark_noise.data**2 + sky_noise.data**2 + source_noise.data**2)
        snr = spec.copy()
        snr.data = nph_source.data / tot_noise.data 
        res = {}
        res['cube'] = {}
        res['input'] = dict(atm_abs=ins_atm, ins_trans=ins_ins, atm_emi=ins_sky, flux_source=spec)
        res['spec'] = dict(snr=snr,
                           snr_mean=snr.data.mean(),
                           snr_max=snr.data.max(),
                           snr_min=snr.data.min(),
                           nph_source=nph_source,
                           nph_sky=nph_sky,
                           frac_ima=frac_ima, 
                           nb_spaxels=nspaxels, 
                           nb_voxels=nb_voxels,
                           nb_spectels=1,                           
                           noise = dict(ron=ron_noise, dark=dark_noise, sky=sky_noise, source=source_noise, tot=tot_noise),   
                           )        
        return res     
            
    def flux_from_source(self, ins, snr, ima, spec, waves=None, flux=None, bracket=(0.1,100000)):
        """compute the flux needed to achieve a given S/N

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        flux : float
            flux value in erg/s/cm2 (for line), in erg/s/cm2/A (for cont), / arcsec2 (for sb)
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum                   
        waves : tuple
             wavelength range to compute the S/N for cont source (Default value = None)
        flux :
             starting value of the flux (Default value = None)
        bracket : tuple of float
             interval of flux*1.e-20 for the zero finding routine (Default value = (0.1,100000) :
            

        Returns
        -------
        dict
            result dictionary (see documentation)

        """
        global g_frac_ima,g_size_ima,g_nspaxels
        if waves is None:
            krange = None
        else:
            krange = spec.wave.pixel(waves, nearest=True)
            krange[1] += 1            
        # compute frac_ima and nspaxels only once
        if flux is None:
            flux = 1.e-18
        if (self.obs['ima_type'] == 'ps') and (self.obs['spec_type']=='cont'):
            lbin = 20 if spec.shape[0]>100 else 1
            g_frac_ima,g_size_ima,g_nspaxels = self.get_psf_frac_ima(ins, flux, spec, lbin=lbin)
        res0 = root_scalar(self.fun, args=(snr, ins, ima, spec, krange), 
                           method='brenth', bracket=bracket, xtol=1.e-3, maxiter=100)   
        flux = res0.root*1.e-20
        res = self.snr_from_source(ins, flux, ima, spec, waves)
        if krange is not None:
            snr1 = np.mean(res['spec']['snr'][krange[0]:krange[1]].data)
            res['spec']['snr_mean'] = snr1
            res['spec']['flux'] = flux
        else:
            snr1 = res['aper']['snr']
            res['aper']['flux'] = flux
        self.logger.debug('SN %.2f Flux %.2e Iter %d Fcall %d converged %s', snr1, flux, res0.iterations, 
                          res0.function_calls, res0.converged)
        g_frac_ima,g_size_ima,g_nspaxels = None,None,None

        return res 
    
    def fun(self, flux, snr0, ins, ima, spec, krange):
        """ minimizing function used by flux_from_source

        Parameters
        ----------
        flux : float
            flux value * 1.e-20
        snr0 : float
            target S/N
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        krange : tuple of int
            wavelength range in spectel to compute the S/N

        Returns
        -------
        float
            S/N - target S/N
        """        
        res = self.snr_from_source(ins, flux*1.e-20, ima, spec, loop=True, debug=False)
        if krange is not None:
            snr = np.mean(res['spec']['snr'][krange[0]:krange[1]].data)
        else:
            snr = res['aper']['snr']
        #print(f"flux {flux:.2f} snr {snr:.3f} snr0 {snr0:.1f} diff {snr-snr0:.5f}")
        return snr-snr0      
        
    def get_image_quality(self, ins):
        """ compute image quality evolution with wavelength

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])

        Returns
        -------
        numpy array of float
            image quality 

        """
        obs = self.obs
        iq = ins['instrans'].copy()
        waves = iq.wave.coord()
        iq.data = get_seeing_fwhm(obs['seeing'], obs['airmass'], waves, self.tel['diameter'], ins['iq_fwhm'])
        return iq
        
    def get_image_psf(self, ins, wave, oversamp=10):
        """ compute PSF image

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        wave : float
            wavelength in A
        oversamp : int
             oversampling factor (Default value = 10)

        Returns
        -------
        MPDAF image
            PSF image

        """
        fwhm = get_seeing_fwhm(self.obs['seeing'], self.obs['airmass'], wave, 
                               self.tel['diameter'], ins['iq_fwhm'])
        ima = moffat(ins['spaxel_size'], fwhm, ins['iq_beta'], oversamp=oversamp) 
        ima.oversamp = oversamp 
        return ima
    
    def print_aper(self, res, names):
        """ pretty print the apertures results for a set of results
   
        Parameters
        ----------
        res : dict or list of dict
            result dictionaries deliver by snr_from_source or flux_from_source
        names : str or list of str
            name to identify the result
   
        Returns
        -------
        astropy table
            table with one column by result
        """        
        if not isinstance(res, (list)):
            res,names = [res],[names]
        tab = Table(names=['item']+names, dtype=(len(res)+1)*['S20'])
        
        for key in res[0]['aper'].keys():
            d = dict(item=key)
            if isinstance(res[0]['aper'][key], (float, np.float)):
                for n,r in zip(names,res):
                    d[n] = f"{r['aper'][key]:5.4g}"
            else: 
                for n,r in zip(names,res):
                    d[n] = f"{r['aper'][key]}"
            tab.add_row(d)
        return tab    

        
def _copy_obs(obs, res):
    """ copy obs dict in res dict """
    for key,val in obs.items():
        res[key] = val
        
def asymgauss(ftot, l0, sigma, skew, wave):
    """compute asymetric gaussian

    Parameters
    ----------
    ftot : float
        total flux
    l0 : float
        peak wavelength in A
    sigma : float
        sigma in A
    skew : float
        skew parameter (0 for a gaussian)
    wave : numpy array of float
        wavelengths in A

    Returns
    -------
    numpy array of float
        asymetric gaussian values

    """
    dl = wave - l0
    g = np.exp(-dl**2/(2*sigma**2))
    f = 1 + erf(skew*dl/(1.4142135623730951*sigma))
    h = f*g
    h = ftot * h/h.sum()
    return h 

def peakwave_asymgauss(lpeak, sigma, skew, dl=0.01):
    """ compute the asymetric gaussian wavelength paramater to get the given peak wavelength

    Parameters
    ----------
    lpeak : float
        peak wavelength in A
    sigma : float
        sigma in A
    skew : float
        skew parameter (0 for a gaussian)        
    dl : float
       step in wavelength (A, Default value = 0.01)

    Returns
    -------
    float
        wavelength (A) of the asymetric gaussian

     """
    wave = np.arange(lpeak-5*sigma,lpeak+5*sigma,dl)
    res0 = root_scalar(_funasym, args=(lpeak, sigma, skew, wave), method='brentq',
                          bracket=[lpeak-2*sigma,lpeak+2*sigma], rtol=1.e-3, maxiter=100) 
    return res0.root

def _funasym(l0, lpeak, sigma, skew, wave):
    """ function used to minimize in peakwave_asymgauss """
    f = asymgauss(1.0, l0, sigma, skew, wave)
    k = f.argmax()
    zero = wave[k] - lpeak
    #print(zero)
    return zero

def _fun_aper(kfwhm, obj, ins, flux, ima, spec, krange=None):
    """ function used to minimize in optimum_circular_aperture """
    obj.obs['ima_kfwhm'] = kfwhm
    res = obj.snr_from_source(ins, flux, ima, spec, debug=False)
    if krange is None:
        snr = res['aper']['snr']
    else:
        snr = np.mean(res['spec']['snr'].data[krange[0]:krange[1]])
    return -snr    

def _fun_range(kfwhm, obj, ins, flux, ima, spec):
    """ function used to minimize in optimum_spectral_range """
    obj.obs['spec_range_kfwhm'] = kfwhm
    res = obj.snr_from_source(ins, flux, ima, spec, debug=False)
    snr = res['aper']['snr']
    #print(kfwhm, snr)
    return -snr    

def vdisp2sigma(vdisp, l0):
    """compute sigma in A from velocity dispersion in km/s

    Parameters
    ----------
    vdisp : float
        velocity dispersion (km/s)
    l0 : float
        wavlenegth (A)

    Returns
    -------
    float
        sigma in A

       """
    return vdisp*l0/C_kms

def sigma2vdisp(sigma, l0):
    """compute sigma in A from velocity dispersion in km/s

    Parameters
    ----------
    sigma : float
        sigma in A
    l0 : float
        wavelength in A

    Returns
    -------
    float
        velocity dispersion in km/s

     """
    return sigma*C_kms/l0

def fwhm_asymgauss(lbda, flux):
    """ compute the FWHM of an asymmetric gaussian 

    Parameters
    ----------
    lbda : numpy array of float
        wavelength array in A
    flux : numpy array of float
        asymetric gaussian values

    Returns
    -------
    tuple
        l0,l1,l2
        l0 peak wavelength (A)
        l1 blue wavelength at FWHM (A)
        l2 red wavelength at FWHM (A)

    """    
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

def moffat(samp, fwhm, beta, ell=0, kfwhm=2, oversamp=1, uneven=1):
    """ compute a 2D Moffat image

    Parameters
    ----------
    samp : float
        image sampling in arcsec
    fwhm : float
        FWHM of the MOFFAT (arcsec)
    beta : float
        MOFFAT shape parameter (beta > 4 for Gaussian, 1 for Lorentzien)
    ell : float
         image ellipticity (Default value = 0)
    kfwhm : float
         factor relative to the FWHM to compute the size of the image (Default value = 2)
    oversamp : int
         oversampling gfactor (Default value = 1)
    uneven : int
         if 1 the image size will have an uneven number of spaxels (Default value = 1)

    Returns
    -------
    MPDAF image
         MOFFAT image
    """    
    ns = (int((kfwhm*fwhm/samp+1)/2)*2 + uneven)*oversamp
    pixfwhm = oversamp*fwhm/samp
    pixfwhm2 = pixfwhm*(1-ell)
    ima = moffat_image(fwhm=(pixfwhm2,pixfwhm), n=beta, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.data /= ima.data.sum()
    return ima

def compute_sky(lbda1, lbda2, dlbda, lsf, moon, airmass=1.0):
    """ compute Paranal sky model from ESO skycalc

    Parameters
    ----------
    lbda1 : float
        starting wavelength (A)
    lbda2 : float
        ending wavelength (A)
    dlbda : float
        step in wavelength (A)
    lsf : float
        LSF size in spectels
    moon : str
        moon brightness (eg darksky)
    airmass : float
         observation airmass (Default value = 1.0)

    Returns
    -------
    astropy table
        sky table as computed by skycalc

    """
    skyModel = SkyModel()
    if moon == 'darksky':
        sep = 0
    elif moon == 'greysky':
        sep = 90
    elif moon == 'brightsky':
        sep = 180   
    else:
        raise ValueError(f'Error in moon {moon}')
    skypar = dict(wmin=lbda1*0.1, wmax=lbda2*0.1, wdelta=dlbda*0.1, 
                  lsf_gauss_fwhm=lsf, lsf_type='Gaussian', airmass=airmass,
                  moon_sun_sep=sep, observatory='paranal')
    skypar = fixObservatory(skypar)
    skyModel.callwith(skypar)
    f = BytesIO()
    f = BytesIO(skyModel.data)
    tab = Table.read(f)
    tab.iden = f"{moon}_{airmass:.1f}"
    return tab

def show_noise(r, ax, legend=False, title='Noise fraction'):
    """ plot the noise characteristics from the result dictionary

    Parameters
    ----------
    r : dict
          result dictionary that contain the noise results       
    ax : amtplolib axis
          axis where to plot
    legend : bool
         if True display legend on the figure (Default value = False)
    title : str
         title to display (Default value = 'Noise fraction')

    """    
    rtot = r['tot']
    f = (r['sky'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='r', label='sky' if legend else None)
    f = (r['source'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='b', label='source' if legend else None)
    f = (r['ron']/rtot.data)**2 if isinstance(r['ron'],float) else (r['ron'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='g', label='ron' if legend else None)
    f = (r['dark']/rtot.data)**2 if isinstance(r['dark'],float) else (r['dark'].data/rtot.data)**2
    ax.plot(rtot.wave.coord(), f, color='m', label='dark' if legend else None)
    if isinstance(r['ron'],float):
        f = (r['ron']**2 + r['dark']**2)/rtot.data**2
    else:
        f = (r['ron'].data**2 + r['dark'].data**2)/rtot.data**2
    ax.plot(rtot.wave.coord(), f, color='k', label='ron+dark' if legend else None)
    if legend:
        ax.legend(loc='upper right')
    ax.axhline(0.5, color='k', alpha=0.2)
    ax.axhline(0.5, color='k')
    ax.set_title(title)
        

def get_data(obj, chan, name, refdir): 
    """ retreive instrument data from the associated setup files

    Parameters
    ----------
    obj : ETC class 
        instrument class (e.g. etc.ifs)
    chan : str
        channel name (eg 'red')
    name : str
        instrument name (eg 'ifs')
    refdir : str
        directory path where the setup fits file can be found

    """    
    ins = obj[chan]       
    ins['wave'] = WaveCoord(cdelt=ins['dlbda'], crval=ins['lbda1'], cunit=u.angstrom)
                            
    flist = glob.glob(os.path.join(refdir,f"{name}_{chan}_*sky_*.fits"))
    flist.sort()
    ins['sky'] =[]
    moons = []
    for fname in flist:
        f = os.path.basename(fname).split('_')
        moon = f[2]
        moons.append(moon)
        airmass = float(f[3][:-5])
        d = dict(moon=moon, airmass=airmass)
        tab = Table.read(fname)
        for key,val in [['LSFPIX',ins['lsfpix']],
                         ['LBDA1',ins['lbda1']],
                         ['LBDA2',ins['lbda2']],
                         ['DLBDA',ins['dlbda']],
                         ['MOON', moon],
                         ['AIRMASS', airmass]
                         ]:
            if key in tab.meta:
                if isinstance(tab.meta[key], float):
                    if not np.isclose(tab.meta[key], val):
                        raise ValueError(f"Incompatible {key} values between {fname} and setup")
                else:
                    if tab.meta[key] != val:
                        raise ValueError(f"Incompatible {key} values between {fname} and setup")
        if abs(tab['lam'][0]*10 - ins['lbda1'])>ins['dlbda'] or \
           abs(tab['lam'][-1]*10 - ins['lbda2'])>ins['dlbda'] or \
           abs(tab['lam'][1]-tab['lam'][0])*10 - ins['dlbda']>0.01:
            raise ValueError(f'Incompatible bounds between {fname} and setup') 
        d['emi'] = Spectrum(data=tab['flux'], wave=ins['wave'])
        d['abs'] = Spectrum(data=tab['trans'], wave=ins['wave'])
        d['filename'] = fname
        ins['sky'].append(d) 
    filename = f'{name}_{chan}_noatm.fits'
    trans=Table.read(os.path.join(refdir,filename))
    if trans['WAVE'][0]*10 > ins['lbda1'] or \
       trans['WAVE'][-1]*10 < ins['lbda2'] :
        raise ValueError(f'Incompatible bounds between {filename} and setup')        
    ins['instrans'] = Spectrum(data=np.interp(ins['sky'][0]['emi'].wave.coord(),trans['WAVE']*10,trans['TOT']), 
                                                wave=ins['sky'][0]['emi'].wave) 
    ins['instrans'].filename = filename
    ins['skys'] = list(set(moons))
    ins['wave'] = ins['instrans'].wave
    ins['chan'] = chan
    ins['name'] = name
    return

def update_skytables(logger, obj, name, chan, moons, airmass, refdir, overwrite=False, debug=False):
    """ update setup sky files for a change in setup parameters

    Parameters
    ----------
    logger : logging instance
        logger to print progress
    obj : dict
        instrument dictionary
    name : str
        instrument name
    chan: str
        channel name
    moons : list of str
        list of moon sky conditions eg ['darksky']
    airmass : list of float
        list of airmass
    refdir : str
        path name where to write the reference setup fits file
    overwrite : bool
         if True overwrite existing file (Default value = False)
    debug : bool
         if True do not try to write(used for unit test, Default value = False)

    """
    for am in airmass:
        for moon in moons:
            tab = compute_sky(obj['lbda1'], obj['lbda2'], obj['dlbda'], 
                              obj['lsfpix'], moon, am)
            tab.meta['lsfpix'] = obj['lsfpix']
            tab.meta['moon'] = moon
            tab.meta['airmass'] = am
            tab.meta['lbda1'] = obj['lbda1']
            tab.meta['lbda2'] = obj['lbda2']
            tab.meta['dlbda'] = obj['dlbda']
            fname = f"{name}_{chan}_{moon}_{am:.1f}.fits"
            filename = os.path.join(refdir, fname)
            logger.info('Updating file %s', filename)
            if debug:
                logger.info('Debug mode table not saved to file')
            else:
                tab.write(filename, overwrite=overwrite)
            
def get_seeing_fwhm(seeing, airmass, wave, diam, iq_ins):
    """ compute FWHM for the Paranal ESO ETC model

    Parameters
    ----------
    seeing : float
        seeing (arcsec) at 5000A
    airmass : float
        airmass of the observation
    wave : numpy array of float
        wavelengths in A
    diam : float
        telescope primary mirror diameter in m
    iq_ins : float of numpy array
        image quality of the telescope + instrument

    Returns
    -------
    numpy array of float
        FWHM (arcsec) as function of wavelengths

    """
    r0,l0 = 0.188,46 # for VLT (in ETC)
    Fkolb = 1/(1+300*diam/l0)-1
    iq_atm = seeing*(wave/5000)**(-1/5)*airmass**(3/5) * \
        np.sqrt(1+Fkolb*2.183*(r0/l0)**0.356)
    iq = np.sqrt(iq_atm**2 + iq_ins**2)
    return iq

def _checkobs(obs, saved, keys):
    """ check existence and copy keywords from obs to saved """
    for key in keys:
        if key not in obs.keys():
            raise KeyError(f'keyword {key} missing in obs dictionary')
        saved[key] = obs[key]
        
   