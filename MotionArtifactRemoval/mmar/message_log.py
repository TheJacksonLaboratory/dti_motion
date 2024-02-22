# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
This script implements a simple message logging class.

Example - No immediate output; accumulate messages until save.
    log = MessageLog()
    ...
    log.add_msg(s)
    ...
    log.save_log(file)

Example - print messages immediately while accumulating messages till save
    log = MessageLog(print)
    ...
    log.add_msg(s)
    ...
    log.save_log(file)


To Do:
    - check for exceptions when writing file

'''

import os

def no_print(_str):
    ''' Empty print function '''
    return True

class MessageLog:
    ''' Class for logging messages '''

    cumulative_message = ""
    print_func = print


    def __init__(self, display_function=no_print):
        self.print_func = display_function

    def add_msg(self, msg):
        '''
        Adds message line to cumulative message string.

        Parameters:
            msg : str
                message to add to the cumulative log
        Return:
            None
        '''
        self.cumulative_message += msg
        self.cumulative_message += os.linesep
        self.print_func(msg)


    def save_log(self, file_path):
        '''
        Write the cumulative log to the specified file.

        Parameters:
            file_path : str
                The full file path of the log file to write.
        Return:
            None
        '''

        with open(file_path, 'w', encoding="latin-1") as file:
            file.write(self.cumulative_message)
