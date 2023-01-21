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
                        diam = 8.20, # pirmary diameter in meter
                        )      
        # IFS
        self.ifs = {} # 
        self.ifs['channels'] = ['blue','red']
        
        # IFS BlueMUSE
        chan = 'blue'
        self.ifs[chan] = dict(desc = 'BlueMUSE 5/01/2022',
                                name = 'ifs',
                                chan = chan, 
                                type='IFS',
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
                               name = 'ifs',
                               chan = chan, 
                               type='IFS',
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
                                  name = 'giraffe',
                                  chan = chan,                                   
                                  type = 'MOS',
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
        self.logger.info('%s ETC version: %s', self.name, self.version)
        self.logger.debug('Area: %.1f m2', self.tel['area'])
        for ins_name in ['ifs', 'giraffe']: 
            insfam = getattr(self, ins_name)
            for chan in insfam['channels']:
                ins = insfam[chan]
                self.logger.debug('%s type %s Channel %s', ins_name.upper(), ins['type'], chan)
                
                self.logger.debug('\t %s', ins['desc'])
                self.logger.debug('\t Spaxel size: %.2f arcsec', ins['spaxel_size'])
                self.logger.debug('\t Wavelength range %s A step %.2f A LSF %.1f pix', ins['instrans'].get_range(), ins['instrans'].get_step(), ins['lsfpix'])
                self.logger.debug('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                                  ins['instrans'].data.max(), ins['instrans'].wave.coord(ins['instrans'].data.argmax()),
                                  ins['instrans'].data.min(), ins['instrans'].wave.coord(ins['instrans'].data.argmin()))
                self.logger.debug('\t Detector RON %.1f e- Dark %.1f e-/h', ins['ron'],ins['dcurrent'])
                for sky in ins['sky']:
                    self.logger.debug('\t Sky moon %s airmass %s table %s', sky['moon'], sky['airmass'], 
                                      os.path.basename(sky['filename']))
                self.logger.debug('\t Instrument transmission table %s', os.path.basename(ins['instrans'].filename))          
          
                        
        
