import logging
import os

class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(self.log_file_path)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def write(self, message):
        self.logger.info(message)

    @staticmethod
    def make_log_filename(fname):
        return os.path.join(os.path.dirname(fname), 'log_'+os.path.splitext(os.path.basename(fname))[0]+'.txt')

if __name__ == '__main__':
    log_file_path = os.path.join(os.path.dirname(__file__), 'log.txt')
    logger = Logger(log_file_path)
    logger.write('test')