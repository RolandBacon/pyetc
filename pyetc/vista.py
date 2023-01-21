import logging
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
import astropy.units as u
from .etc import ETC


CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/VISTA')

class VISTA(ETC):
    
    def __init__(self, log=logging.INFO):
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        # Telescope
        self.name = 'VISTA'
        self.tel = dict(area = np.pi*3.7**2,# effective aperture in squared meter (TBC)
                        diam = 4.0, # pirmary diameter in meter
                        )  
        # --------- MOSLR -------------
        self.moslr = {} 
        self.moslr['channels'] = ['blue','green','red']
        self.moslr['skys'] = ['darksky','greysky']        
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                type = 'MOS',
                                spaxel_size = 0.40, # spaxel size in arcsec TBC
                                aperture = 1.45, # fiber diameter in arcsec
                                dlbda = 0.53, # Angstroem/pixel TBC should be 0.31 ?
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 5440, # end wavelength in Angstroem
                                lsfpix = 2.5, # LSF in spectel TBC
                                ron = 2.4, # readout noise (e-)
                                dcurrent = 1.94, # dark current (e-/pixel/h)                                
                                )
        _get_data(self.moslr, chan, 'moslr')        
        # MOS-LR green channel      
        chan = self.moslr['channels'][1] 
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                type = 'MOS',
                                spaxel_size = 0.25, # spaxel size in arcsec
                                aperture = 1.605, # fiber diameter in arcsec
                                dlbda = 0.46, # Angstroem/pixel
                                lbda1 = 5250, # starting wavelength in Angstroem
                                lbda2 = 7100, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )     
        _get_data(self.moslr, chan, 'moslr')  
        # MOS-LR red channel      
        chan = self.moslr['channels'][2]
        self.moslr[chan] = dict(desc='Based on 4MOST LR throughput 5/01/2022',
                                type = 'MOS',
                                aperture = 1.0, # fiber diameter in arcsec
                                spaxel_size = 0.25, # spaxel size in arcsec
                                dlbda = 0.53, # Angstroem/pixel
                                lbda1 = 7000, # starting wavelength in Angstroem
                                lbda2 = 9150, # end wavelength in Angstroem
                                lsfpix = 4.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )        
        _get_data(self.moslr, chan, 'moslr')           
        
    def info(self):
        self.logger.info('%s ETC version: %s', self.name, self.version)
        self.logger.debug('Area: %.1f m2', self.tel['area'])      
            
        for chan in self.moslr['channels']:
            self.logger.debug('MOS-LR Channel %s', chan)
            mos = self.moslr[chan]
            self.logger.debug('\t %s', mos['desc'])
            self.logger.debug('\t Spaxel size: %.2f arcsec', mos['spaxel_size'])
            self.logger.debug('\t Fiber aperture: %.1f arcsec', mos['aperture'])
            self.logger.debug('\t Wavelength range %s A step %.2f A LSF %.1f pix', mos['instrans'].get_range(), mos['instrans'].get_step(), mos['lsfpix'])
            self.logger.debug('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                              mos['instrans'].data.max(), mos['instrans'].wave.coord(mos['instrans'].data.argmax()),
                              mos['instrans'].data.min(), mos['instrans'].wave.coord(mos['instrans'].data.argmin()))
            self.logger.debug('\t Detector RON %.1f e- Dark %.1f e-/h', mos['ron'],mos['dcurrent'])
            self.logger.debug('\t Atmospheric transmission table %s', mos['atmtrans'].filename)
            self.logger.debug('\t Dark sky emission table %s', mos['darksky'].filename)
            self.logger.debug('\t Grey sky emission table %s', mos['greysky'].filename)
            self.logger.debug('\t Instrument transmission table %s', mos['instrans'].filename)        
            
            
def _get_data(obj, chan, name): 
    ins = obj[chan]       
    ins['wave'] = WaveCoord(cdelt=ins['dlbda'], crval=ins['lbda1'], 
                            shape=int((ins['lbda2']-ins['lbda1'])/ins['dlbda']+0.5),
                            cunit=u.angstrom)
    filename = f'{name}_skytable_moon05_am1_lsfconv_{chan}.fits'
    skytab_grey = Table.read(os.path.join(CURDIR,filename))
    if abs(skytab_grey['lam'][0]*10 - ins['lbda1'])>ins['dlbda'] or \
       abs(skytab_grey['lam'][-1]*10 - ins['lbda2'])>ins['dlbda'] or \
       abs(skytab_grey['lam'][1]-skytab_grey['lam'][0])*10 - ins['dlbda']>0.01:
        raise ValueError(f'Incompatible bounds between {filename} and setup')
    ins['atmtrans'] = Spectrum(data=skytab_grey['trans'], wave=ins['wave'])
    ins['atmtrans'].filename = filename
    ins['greysky'] = Spectrum(data=skytab_grey['flux'], wave=ins['wave'])
    ins['greysky'].filename = filename
    filename = f'{name}_skytable_newmoon_am1_lsfconv_{chan}.fits'
    skytab_dark = Table.read(os.path.join(CURDIR,filename))
    if abs(skytab_dark['lam'][0]*10 - ins['lbda1'])>ins['dlbda'] or \
       abs(skytab_dark['lam'][-1]*10 - ins['lbda2'])>ins['dlbda'] or \
       abs(skytab_dark['lam'][1]-skytab_grey['lam'][0])*10 - ins['dlbda']>0.01: 
        raise ValueError(f'Incompatible bounds between {filename} and setup')  
    ins['darksky'] = Spectrum(data=skytab_dark['flux'], wave=ins['wave'])
    ins['darksky'].filename = filename
    filename = f'{name}_{chan}_noatm.fits'
    trans=Table.read(os.path.join(CURDIR,filename))
    if trans['WAVE'][0]*10 > ins['lbda1'] or \
       trans['WAVE'][-1]*10 < ins['lbda2'] :
        raise ValueError(f'Incompatible bounds between {filename} and setup')        
    ins['instrans'] = Spectrum(data=np.interp(ins['atmtrans'].wave.coord(),trans['WAVE']*10,trans['TOT']), 
                                            wave=ins['atmtrans'].wave) 
    ins['instrans'].filename = filename
