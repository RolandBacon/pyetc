# Test main

from pyetc.etc import vdisp2sigma
from pyetc.wst import WST
from mpdaf.obj import Spectrum, WaveCoord, mag2flux, flux2mag
import numpy as np
from matplotlib import pyplot as plt



print('start')
wst = WST(log='DEBUG')

wst.logger.debug('debug mode to be seen')
wst.set_logging("INFO")
wst.logger.debug('debug mode not to be seen')
wst.logger.info('info mode to be seen')

print('end')
