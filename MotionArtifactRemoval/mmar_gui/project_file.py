# Copyright The Jackson Laboratory, 2022
# authors: Jim Peterson, Abed Ghanbari
'''
Implimentation of the MmarProjectFile class.
'''
import os
import configparser

class MmarProjectFile():
    '''
    This class implements saving and loading project/config files for the mmar project.
    '''

    def save(self, settings_dict, filename):
        '''
        Saves the project settings to the specified file.
        Parameters:
            settings_dict : dictinary
                dictionary containing all settings
            filename : string
                full pathname of file to save
        Return:
            none
        '''
        if not filename:
            return

        config = configparser.ConfigParser()
        config['settings'] = settings_dict
        with open(filename, 'w', encoding="latin-1") as configfile:
            config.write(configfile)


    def load(self, filename):
        '''
        Loads the specified project file.
        Any missing fields are initialized with default values.
        Parameters:
            filename : string
                name of full pathname of file to load
        '''
        settings_tmp = None
        if filename and os.path.exists(filename):
            config = configparser.ConfigParser()
            config.read(filename)
            settings_tmp = config['settings']
        if settings_tmp == None:
            settings_tmp = {}

        settings = {}

        if settings_tmp.get('input_dir'):
            settings['input_dir'] = settings_tmp.get('input_dir')
        else:
            settings['input_dir'] = "sample_data"


        if settings_tmp.get('output_dir'):
            settings['output_dir'] = settings_tmp.get('output_dir')
        else:
            settings['output_dir'] = "results"

        if settings_tmp.get('model_file'):
            settings['model_file'] = settings_tmp.get('model_file')
        else:
            settings['model_file'] = ""

        if (settings_tmp.get('use_alt_model') != None) and (settings_tmp.get('use_alt_model') == 'True') and (settings['model_file']):
            settings['use_alt_model'] = True
        else:
            settings['use_alt_model'] = False

        if settings_tmp.get('ml_prob') == None:
            settings['ml_prob'] = 0.5
        else:
            settings['ml_prob'] = float(settings_tmp['ml_prob'])

        if settings_tmp.get('ml_classifier'):
            settings['ml_classifier'] = settings_tmp.get('ml_classifier')
        else:
            settings['ml_classifier'] = 'Random Forest'

        if settings_tmp.get('image_filename'):
            settings['image_filename'] = settings_tmp.get('image_filename')
        else:
            settings['image_filename'] = "dti/image.nii.gz"

        if settings_tmp.get('use_metadata') and (settings_tmp.get('use_metadata') == 'False'):
            settings['use_metadata'] = False
        else:
            settings['use_metadata'] = True

        if settings_tmp.get('bval_filename'):
            settings['bval_filename'] = settings_tmp.get('bval_filename')
        else:
            settings['bval_filename'] = "bvals.txt"

        if settings_tmp.get('bvec_filename'):
            settings['bvec_filename'] = settings_tmp.get('bvec_filename')
        else:
            settings['bvec_filename'] = "bvecs.txt"

        if settings_tmp.get('use_subdirs') and (settings_tmp.get('use_subdirs') == 'True'):
            settings['use_subdirs'] = True
        else:
            settings['use_subdirs'] = False

        if settings_tmp.get('save_4d') and (settings_tmp.get('save_4d') == 'True'):
            settings['save_4d'] = True
        else:
            settings['save_4d'] = False

        return settings

