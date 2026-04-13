import logging

logger = logging.getLogger('klanker.logger')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
logger.addHandler(handler)