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
        self.ifs[chan] = dict(desc = 'Inspired from BlueMUSE throughput',
                              version = '0.1 10/02/2023',
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
        self.ifs[chan] = dict(desc='Inspired from MUSE throughput', 
                               version = '0.1 10/02/2023',
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
              
        # # --------- MOSLR-VIS 2 channels 6k CCD -------------
        self.moslr = {} 
        self.moslr['channels'] = ['blue','red']       
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput', 
                                version = '0.1 10/02/2023',
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
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput',
                                version = '0.1 10/02/2023',
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
            
        # --------- MOSLR-VIS+IR 4 channels 4k CCD & GECCD -------------
        self.moslr2 = {} 
        self.moslr2['channels'] = ['blue','green','red','ir']       
        # MOS-LR blue channel 
        chan = self.moslr2  ['channels'][0]
        self.moslr2[chan] = dict(desc='Inspired from 4MOST LR throughput',  
                                version = '0.1 10/02/2023',
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.375, # Angstroem/pixel
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 5200, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr2, chan, 'moslr2', CURDIR)        
        # MOS-LR green channel      
        chan = self.moslr2['channels'][1] 
        self.moslr2[chan] = dict(desc='Inspired from 4MOST LR throughput',
                                version = '0.1 10/02/2023',
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.50, # Angstroem/pixel
                                lbda1 = 5100, # starting wavelength in Angstroem
                                lbda2 = 7100, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr2, chan, 'moslr2', CURDIR) 
        # MOS-LR red channel      
        chan = self.moslr2['channels'][2] 
        self.moslr2[chan] = dict(desc='Inspired from 4MOST LR throughput',
                                version = '0.1 10/02/2023',
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.675, # Angstroem/pixel
                                lbda1 = 7000, # starting wavelength in Angstroem
                                lbda2 = 9700, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr2, chan, 'moslr2', CURDIR)                      
        # MOS-LR ir channel      
        chan = self.moslr2['channels'][3] 
        self.moslr2[chan] = dict(desc='First guess 31/01/2023',
                                version = '0.1 10/02/2023',
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.85, # Angstroem/pixel
                                lbda1 = 9600, # starting wavelength in Angstroem
                                lbda2 = 13000, # end wavelength in Angstroem
                                lsfpix = 4.1, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr2, chan, 'moslr2', CURDIR)            
        
    def info(self):
        self._info(['ifs', 'moslr', 'moslr2'])
                
           
            

               
        
        