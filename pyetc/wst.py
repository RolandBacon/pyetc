import logging
from pyetc.version import  __version__
from mpdaf.obj import Spectrum, WaveCoord
from astropy.table import Table
import os, sys
import numpy as np
from mpdaf.log import setup_logging
import astropy.units as u
from .etc import ETC


CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/WST')

class WST(ETC):
    
    def __init__(self, log=logging.INFO):
        self.version = __version__
        self.logger = logging.getLogger(__name__)
        setup_logging(__name__, level=log, stream=sys.stdout)        
        # Telescope
        self.name = 'WST'
        self.tel = dict(area = 100.0) # WST squared meter 
        # IFS
        self.ifs = {} 
        self.ifs['channels'] = ['blue','red']
        self.ifs['skys'] = ['darksky','greysky']
        # IFS blue channel
        self.ifs['blue'] = dict(spaxel_size = 0.25, # spaxel size in arcsec
                                dlbda = 0.60, # Angstroem/pixel
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 5930, # end wavelength in Angstroem
                                lsfpix = 2.5, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        self.ifs['blue']['wave'] = WaveCoord(cdelt=self.ifs['blue']['dlbda'], 
                                             crval=self.ifs['blue']['lbda1'], 
                                             cunit=u.angstrom)
        filename = 'skytable_moon05_am1_lsfconv_blue.fits'
        skytab_grey_blue = Table.read(os.path.join(CURDIR,filename))
        self.ifs['blue']['atmtrans'] = Spectrum(data=skytab_grey_blue['trans'], wave=self.ifs['blue']['wave'])
        self.ifs['blue']['atmtrans'].filename = filename
        self.ifs['blue']['greysky'] = Spectrum(data=skytab_grey_blue['flux'], wave=self.ifs['blue']['wave'])
        self.ifs['blue']['greysky'].filename = filename
        filename = 'skytable_newmoon_am1_lsfconv_blue.fits'
        skytab_dark_blue = Table.read(os.path.join(CURDIR,filename))
        self.ifs['blue']['darksky'] = Spectrum(data=skytab_dark_blue['flux'], wave=self.ifs['blue']['wave'])
        self.ifs['blue']['darksky'].filename = filename
        filename = 'ifs_blue_noatm.txt'
        ifs_trans=np.loadtxt(os.path.join(CURDIR,filename))
        self.ifs['blue']['instrans'] = Spectrum(data=np.interp(self.ifs['blue']['atmtrans'].wave.coord(),ifs_trans[:,0]*10.0,ifs_trans[:,1]), 
                                                wave=self.ifs['blue']['atmtrans'].wave) 
        self.ifs['blue']['instrans'].filename = filename
        
        # IFS red channel
        self.ifs['red'] = dict(spaxel_size = 0.25, # spaxel size in arcsec
                               dlbda = 0.93, # Angstroem/pixel
                               lbda1 = 5930, # starting wavelength in Angstroem
                               lbda2 = 9300, # end wavelength in Angstroem
                               lsfpix = 2.5, # LSF in spectel
                               ron = 3.0, # readout noise (e-)
                               dcurrent = 3.0, # dark current (e-/pixel/h)
                               )
        self.ifs['red']['wave'] = WaveCoord(cdelt=self.ifs['red']['dlbda'], 
                                             crval=self.ifs['red']['lbda1'], 
                                             cunit=u.angstrom)
        filename = 'skytable_moon05_am1_lsfconv_red.fits'
        skytab_grey_red = Table.read(os.path.join(CURDIR,filename))
        self.ifs['red']['atmtrans'] = Spectrum(data=skytab_grey_red['trans'], wave=self.ifs['red']['wave'])
        self.ifs['red']['atmtrans'].filename = filename
        self.ifs['red']['greysky'] = Spectrum(data=skytab_grey_red['flux'], wave=self.ifs['red']['wave'])
        self.ifs['red']['greysky'].filename = filename
        filename = 'skytable_newmoon_am1_lsfconv_red.fits'
        skytab_dark_red = Table.read(os.path.join(CURDIR,filename))
        self.ifs['red']['darksky'] = Spectrum(data=skytab_dark_red['flux'], wave=self.ifs['red']['wave'])
        self.ifs['red']['darksky'].filename = filename
        filename = 'ifs_red_noatm.txt'
        ifs_trans=np.loadtxt(os.path.join(CURDIR,filename))
        self.ifs['red']['instrans'] = Spectrum(data=np.interp(self.ifs['red']['atmtrans'].wave.coord(),ifs_trans[:,0]*10.0,(ifs_trans[:,1]/100)/0.8), 
                                                wave=self.ifs['red']['atmtrans'].wave)
        self.ifs['red']['instrans'].filename = filename
        
        
    def info(self):
        self.logger.info('%s ETC version: %s', self.name, self.version)
        self.logger.debug('Area: %.1f m2', self.tel['area'])
        for chan in ['blue','red']:
            self.logger.debug('IFS Channel %s', chan)
            ifs = self.ifs[chan]
            self.logger.debug('\t Spaxel size: %.2f arcsec', ifs['spaxel_size'])
            self.logger.debug('\t Wavelength range %s A step %.2f A LSF %.1f pix', ifs['instrans'].get_range(), ifs['instrans'].get_step(), ifs['lsfpix'])
            self.logger.debug('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                              ifs['instrans'].data.max(), ifs['instrans'].wave.coord(ifs['instrans'].data.argmax()),
                              ifs['instrans'].data.min(), ifs['instrans'].wave.coord(ifs['instrans'].data.argmin()))
            self.logger.debug('\t Detector RON %.1f e- Dark %.1f e-/h', ifs['ron'],ifs['dcurrent'])
            self.logger.debug('\t Atmospheric transmission table %s', ifs['atmtrans'].filename)
            self.logger.debug('\t Dark sky emission table %s', ifs['darksky'].filename)
            self.logger.debug('\t Grey sky emission table %s', ifs['greysky'].filename)
            self.logger.debug('\t Instrument transmission table %s', ifs['instrans'].filename)
            
                        
        
        
        
        