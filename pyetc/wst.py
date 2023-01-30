import logging
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
import astropy.units as u
from .etc import ETC, get_data


CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/WST')

class WST(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        self.refdir = CURDIR
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        # ------ Telescope ---------
        self.name = 'WST'
        self.tel = dict(area = 100.0,  # squared meter of active area
                        diameter = 11.25 # primary diameter
                        ) 
        # ------- IFS -----------
        self.ifs = {} 
        self.ifs['channels'] = ['blue','red']
        # IFS blue channel
        chan = 'blue'
        self.ifs[chan] = dict(desc = 'Inspired from BlueMUSE throughput 5/01/2022',
                              type = 'IFS',
                              iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                              iq_beta = 2.50, # beta PSF of telescope + instrument
                              spaxel_size = 0.25, # spaxel size in arcsec
                              dlbda = 0.64, # Angstroem/pixel
                              lbda1 = 3700, # starting wavelength in Angstroem
                              lbda2 = 6100, # end wavelength in Angstroem
                              lsfpix = 2.5, # LSF in spectel
                              ron = 3.0, # readout noise (e-)
                              dcurrent = 3.0, # dark current (e-/pixel/h)                                
                              )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', CURDIR)                
        # IFS red channel
        chan = 'red'
        self.ifs[chan] = dict(desc='Inspired from MUSE throughput 5/01/2022',                            
                               type='IFS',
                               iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                               iq_beta = 2.50, # beta PSF of telescope + instrument
                               spaxel_size = 0.25, # spaxel size in arcsec
                               dlbda = 0.97, # Angstroem/pixel
                               lbda1 = 6000, # starting wavelength in Angstroem
                               lbda2 = 9600, # end wavelength in Angstroem
                               lsfpix = 2.5, # LSF in spectel
                               ron = 3.0, # readout noise (e-)
                               dcurrent = 3.0, # dark current (e-/pixel/h)
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', CURDIR)
              
        # --------- MOSLR -------------
        self.moslr = {} 
        self.moslr['channels'] = ['blue','red']       
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput 5/01/2022',                                                             
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.41, # Angstroem/pixel
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 6200, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', CURDIR)        
        # MOS-LR red channel      
        chan = self.moslr['channels'][1] 
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput 5/01/2022',
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.59, # Angstroem/pixel
                                lbda1 = 6000, # starting wavelength in Angstroem
                                lbda2 = 9600, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', CURDIR)  
      
        
    def info(self):
        self._info(['ifs', 'moslr'])
                
           
            

               
        
        