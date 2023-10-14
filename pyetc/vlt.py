import logging
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
import astropy.units as u
from .etc import ETC, get_data


CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/VLT')

class VLT(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        self.refdir = CURDIR
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        # Telescope
        self.name = 'VLT'
        self.tel = dict(area = 48.5,# effective aperture in squared meter 
                        diameter = 8.20, # pirmary diameter in meter
                        )      
        # IFS
        self.ifs = {} # 
        self.ifs['channels'] = ['blue','red']
        
        # IFS BlueMUSE
        chan = 'blue'
        self.ifs[chan] = dict(desc = 'BlueMUSE 5/01/2022',
                                type='IFS',
                                version='1.0', 
                                iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument                                
                                spaxel_size = 0.20, # spaxel size in arcsec
                                dlbda = 0.575, # Angstroem/pixel
                                lbda1 = 3500, # starting wavelength in Angstroem
                                lbda2 = 5800, # end wavelength in Angstroem
                                lsfpix = 2.0, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', CURDIR)
        
        # IFS MUSE
        chan = 'red'
        self.ifs[chan] = dict(desc = 'MUSE 5/01/2022',
                               type='IFS',
                               version='1.0', 
                               iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                               iq_beta = 2.50, # beta PSF of telescope + instrument                               
                               spaxel_size = 0.20, # spaxel size in arcsec
                               dlbda = 1.25, # Angstroem/pixel
                               lbda1 = 4800, # starting wavelength in Angstroem
                               lbda2 = 9300, # end wavelength in Angstroem
                               lsfpix = 2.5, # LSF in spectel
                               ron = 3.0, # readout noise (e-)
                               dcurrent = 3.0, # dark current (e-/pixel/h)
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', CURDIR)   
        
        # MOS Giraffe Medusa
        self.giraffe = {} 
        self.giraffe['channels'] = ['blue']       
        # MOS-LR blue channel 
        chan = self.giraffe['channels'][0]
        self.giraffe[chan] = dict(desc='Based on ESO ETC 17/01/2023',
                                  version='1.0', 
                                  type = 'MOS',
                                  iq_fwhm = 0.40, # fwhm PSF of telescope + instrument
                                  iq_beta = 2.50, # beta PSF of telescope + instrument                                  
                                  spaxel_size = 0.30, # spaxel size in arcsec
                                  aperture = 1.8, # fiber diameter in arcsec
                                  dlbda = 0.12, # Angstroem/pixel
                                  lbda1 = 3702.69, # starting wavelength in Angstroem
                                  lbda2 = 4080.45, # end wavelength in Angstroem
                                  lsfpix = 4.0, # LSF in spectel
                                  ron = 4.0, # readout noise (e-)
                                  dcurrent = 0.5, # dark current (e-/pixel/h)                                
                                ) 
        if not skip_dataload:
            get_data(self.giraffe, 'blue', 'giraffe', CURDIR)                    
        
    def info(self):
        self._info(['ifs', 'giraffe'])        
        
