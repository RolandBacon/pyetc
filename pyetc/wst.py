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

# Cass design BusyWeek Oct 2023
MOS_OBSCURATION = 0.10 # telescope obscuration in the telescope MOS path
IFS_OBSCURATION = 0.18 # telescope obscuration in the telescope IFS path

class WST(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        self.refdir = CURDIR
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        # ------ Telescope ---------
        self.name = 'WST'
        self.tel = dict(area=100.0,  # squared meter of primary mirror (without obscuration)
                        diameter=12.0, # primary diameter 
                        desc='Cass design',
                        version='Busyweek Oct 2023'
                        ) 
        # ------- IFS -----------
        self.ifs = {} 
        self.ifs['channels'] = ['blue','red']
        # IFS blue channel
        chan = 'blue'
        self.ifs[chan] = dict(desc = 'Inspired from BlueMUSE throughput',
                              version = '0.1 10/02/2023',
                              type = 'IFS',
                              obscuration=IFS_OBSCURATION, # IFS telescope obscuration,
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
                               version = '0.2 XX/XX/2023',
                               type='IFS',
                               obscuration=IFS_OBSCURATION, # IFS telescope obscuration,
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
                                version = '0.2 XX/XX/2023',
                                ref = '',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
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
                                ref='',
                                version = '0.1 10/02/2023',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
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
            
        # --------- MOS-HR 4 channels  -------------
        self.moshr = {} 
        self.moshr['channels'] = ['U','B','V','I']       
        # MOS-HR U channel 
        chan = self.moshr['channels'][0]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description',  
                                version = '1.1 13/02/2023',
                                ref = '',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.217, # spaxel size in arcsec
                                aperture = 0.93, # fiber diameter in arcsec
                                dlbda = 0.0217, # Angstroem/pixel
                                lbda1 = 3800, # starting wavelength in Angstroem
                                lbda2 = 4000, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', CURDIR)  
            
        # MOS-HR B channel 
        chan = self.moshr['channels'][1]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description',  
                                version = '1.1 13/02/2023',
                                ref = '',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.207, # spaxel size in arcsec
                                aperture = 0.93, # fiber diameter in arcsec
                                dlbda = 0.029, # Angstroem/pixel
                                lbda1 = 5067, # starting wavelength in Angstroem
                                lbda2 = 5332, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', CURDIR)                    

        # MOS-HR V channel 
        chan = self.moshr['channels'][2]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description',  
                                version = '1.1 13/02/2023',
                                ref = '',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.207, # spaxel size in arcsec
                                aperture = 0.93, # fiber diameter in arcsec
                                dlbda = 0.0367, # Angstroem/pixel
                                lbda1 = 6431, # starting wavelength in Angstroem
                                lbda2 = 6768, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', CURDIR)                    

        # MOS-HR I channel 
        chan = self.moshr['channels'][3]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description',  
                                version = '1.1 13/02/2023',
                                ref = '',
                                type = 'MOS',
                                obscuration=MOS_OBSCURATION, # MOS telescope obscuration,
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.207, # spaxel size in arcsec
                                aperture = 0.93, # fiber diameter in arcsec
                                dlbda = 0.0478, # Angstroem/pixel
                                lbda1 = 8380, # starting wavelength in Angstroem
                                lbda2 = 8820, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', CURDIR)            
        
    def info(self):
        self._info(['ifs', 'moslr', 'moshr'])
                
           
            

               
        
        
