from .wst import *

def _setup_logging():
    import logging
    import sys
    from mpdaf.log import setup_logging
    setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)


_setup_logging()
