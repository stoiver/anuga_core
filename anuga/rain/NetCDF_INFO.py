# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

from anuga.file.netcdf import NetCDFFile
from anuga.config import netcdf_mode_r, netcdf_mode_w, netcdf_mode_a, \
                            netcdf_float
name_in = "gauge_30min_20120228_0000.nc"
infile = NetCDFFile(name_in, netcdf_mode_r)

print infile.variables
sid = infile.variables['station_id']
print sid[:]

prec = infile.variables['precipitation']

import numpy
prec = numpy.zeros((49,1752),numpy.float)
prec.shape
prec[0,:] = infile.variables['precipitation']
print infile.dimensions
#print infile.attributes

infile.close()

