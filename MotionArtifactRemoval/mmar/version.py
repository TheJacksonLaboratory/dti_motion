# Copyright The Jackson Laboratory, 2021
# authors: Jim Peterson, Abed Ghanbari
'''
Simple version class
'''

class Version:
    ''' Constructs and returns the version (e.g. 4.012) '''
    major_ver = 1
    minor_ver = 10

    version_str = "0.000"

    def __init__(self):
        self.version_str = str(self.major_ver) + "." + str(self.minor_ver).zfill(3)

    def str(self):
        ''' returns the string setup in the initialization '''
        return self.version_str

    def major(self):
        ''' returns version major number '''
        return self.major_ver

    def minor(self):
        ''' returns version minor number '''
        return self.minor_ver
