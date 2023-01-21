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
    
    def __init__(self, log=logging.INFO):
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
        self.ifs[chan] = dict(desc = 'Based on BlueMUSE throughput 5/01/2022',
                              name = 'ifs',
                              chan = chan, 
                              type = 'IFS',
                              iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                              iq_beta = 2.50, # beta PSF of telescope + instrument
                              spaxel_size = 0.25, # spaxel size in arcsec
                              dlbda = 0.60, # Angstroem/pixel
                              lbda1 = 3700, # starting wavelength in Angstroem
                              lbda2 = 5930, # end wavelength in Angstroem
                              lsfpix = 2.5, # LSF in spectel
                              ron = 3.0, # readout noise (e-)
                              dcurrent = 3.0, # dark current (e-/pixel/h)                                
                              )
        get_data(self.ifs, chan, 'ifs', CURDIR)                
        # IFS red channel
        chan = 'red'
        self.ifs[chan] = dict(desc='Based on MUSE throughput 5/01/2022',
                               name = 'ifs',
                               chan = chan,                               
                               type='IFS',
                               iq_fwhm = 0.10, # fwhm PSF of telescope + instrument
                               iq_beta = 2.50, # beta PSF of telescope + instrument
                               spaxel_size = 0.25, # spaxel size in arcsec
                               dlbda = 0.93, # Angstroem/pixel
                               lbda1 = 5930, # starting wavelength in Angstroem
                               lbda2 = 9300, # end wavelength in Angstroem
                               lsfpix = 2.5, # LSF in spectel
                               ron = 3.0, # readout noise (e-)
                               dcurrent = 3.0, # dark current (e-/pixel/h)
                               )
        get_data(self.ifs, chan, 'ifs', CURDIR)
              
        # --------- MOSLR -------------
        self.moslr = {} 
        self.moslr['channels'] = ['blue','green','red']       
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                name = 'moslr',
                                chan = chan,                                                               
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.44, # Angstroem/pixel
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 5440, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        get_data(self.moslr, chan, 'moslr', CURDIR)        
        # MOS-LR green channel      
        chan = self.moslr['channels'][1] 
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                type = 'MOS',
                                name = 'moslr',
                                chan = chan,  
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.0, # fiber diameter in arcsec
                                dlbda = 0.46, # Angstroem/pixel
                                lbda1 = 5250, # starting wavelength in Angstroem
                                lbda2 = 7100, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )     
        get_data(self.moslr, chan, 'moslr', CURDIR)  
        # MOS-LR red channel      
        chan = self.moslr['channels'][2]
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                name = 'moslr',
                                chan = chan,                                                               
                                type = 'MOS',
                                iq_fwhm = 0.30, # fwhm PSF of telescope + instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                aperture = 1.0, # fiber diameter in arcsec
                                spaxel_size = 0.25, # spaxel size in arcsec
                                dlbda = 0.53, # Angstroem/pixel
                                lbda1 = 7000, # starting wavelength in Angstroem
                                lbda2 = 9150, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )        
        get_data(self.moslr, chan, 'moslr', CURDIR)           
        
    def info(self):
        self.logger.info('%s ETC version: %s', self.name, self.version)
        self.logger.debug('Area: %.1f m2', self.tel['area'])
        for ins_name in ['ifs', 'moslr']: 
            insfam = getattr(self, ins_name)
            for chan in insfam['channels']:
                ins = insfam[chan]
                self.logger.debug('%s type %s Channel %s', ins_name.upper(), ins['type'], chan)                
                self.logger.debug('\t %s', ins['desc'])
                self.logger.debug('\t Spaxel size: %.2f arcsec Image Quality tel+ins fwhm: %.2f arcsec beta: %.2f ', ins['spaxel_size'], ins['iq_fwhm'], ins['iq_beta'])
                self.logger.debug('\t Wavelength range %s A step %.2f A LSF %.1f pix', ins['instrans'].get_range(), ins['instrans'].get_step(), ins['lsfpix'])
                self.logger.debug('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                                  ins['instrans'].data.max(), ins['instrans'].wave.coord(ins['instrans'].data.argmax()),
                                  ins['instrans'].data.min(), ins['instrans'].wave.coord(ins['instrans'].data.argmin()))
                self.logger.debug('\t Detector RON %.1f e- Dark %.1f e-/h', ins['ron'],ins['dcurrent'])
                for sky in ins['sky']:
                    self.logger.debug('\t Sky moon %s airmass %s table %s', sky['moon'], sky['airmass'], 
                                      os.path.basename(sky['filename']))
                self.logger.debug('\t Instrument transmission table %s', os.path.basename(ins['instrans'].filename))          
                
           
            

               
        
        