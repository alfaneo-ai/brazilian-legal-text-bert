import logging

from sentence_transformers import LoggingHandler


class AppLogger:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
                            handlers=[LoggingHandler()])
        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message)
