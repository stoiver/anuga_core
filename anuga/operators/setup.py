from __future__ import division, print_function

import os
import sys

from os.path import join

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    
    config = Configuration('operators', parent_package, top_path)

    config.add_data_dir('tests')

    #util_dir = os.path.abspath(join(os.path.dirname(__file__),'..','utilities'))
    util_dir = join('..','utilities')
    
    config.add_extension('mannings_operator_ext',
                         sources=['mannings_operator_ext.c'],
                         include_dirs=[util_dir])

    config.add_extension('kinematic_viscosity_operator_ext',
                         sources=['kinematic_viscosity_operator_ext.c'],
                         include_dirs=[util_dir])

    
    return config
    
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
 
