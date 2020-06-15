import numpy
from anuga.utilities import plot_utils as util
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as pyplot

######################################################
# Get ANUGA
p = util.get_output('channel_floodplain1.sww', 0.001)
pc=util.get_centroids(p, velocity_extrapolation=True)

# Indices in the central channel areas
v = (pc.x>15.0)*(pc.x<25.0)

######################################################
# Get hecras info
rasFile='hecras_riverwall_anugaTest/gauges.csv'
rasGauges=numpy.genfromtxt(rasFile,skip_header=3,delimiter=",")

# 24 hrs at 5 minute intervals
rasTime=numpy.linspace(0., 300.*(rasGauges.shape[0]-1), rasGauges.shape[0])

# Get station information for hecras
rasFileO=open(rasFile)
rasStations=rasFileO.readline().split(',')
rasStations[-1]=rasStations[-1].replace('\r\n','')
rasFileO.close()

######################################################

def get_corresponding_series(reach, station):
    """

    Convenience function to get ANUGA/hec-ras series at the same station

    """

    if(reach=='MIDDLE'):
        anuga_x=20.
    elif(reach=='LEFT'):
        anuga_x=35.
    elif(reach=='RIGHT'):
        anuga_x=5.
    else:
        raise Exception('reach not recognized')


    if(not station%100.==0.):
        raise Exception, 'Station must be in 0., 100. , 200., .... 900., 1000.'

    # Get station string in hecras gauges
    if(station>0. and station<1000.):
        station_str=str(int(station))+'.*'
    else:
        station_str=str(int(station))

    ras_string='THREERIVERS '+reach+' '+station_str

    ras_inds=rasStations.index(ras_string)

    ras_data=numpy.vstack([rasTime, rasGauges[:,ras_inds]]).transpose()

    anuga_index=((pc.x-anuga_x)**2+(pc.y-(1000.-station))**2).argmin()

    anuga_data=numpy.vstack([pc.time, pc.stage[:,anuga_index]]).transpose()

    return [ras_data, anuga_data]

def compare_reach(reach):
    #stations=[0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]
    # Skip stations right on the boundary because ANUGA / hecras boundary conditions differ
    stations=[100., 200., 300., 400., 500., 600., 700., 800., 900.]
    stations.reverse() # This looks nicer
    colz=['red', 'blue', 'green', 'purple', 'orange', 'brown', 'black', 'pink', 'yellow']

    for i, station in enumerate(stations):
        try:
            x=get_corresponding_series(reach, station)
            pyplot.plot(x[0][:,0],x[0][:,1],'-', label=str(station), color=colz[i])
            pyplot.plot(x[1][:,0],x[1][:,1],'--', color=colz[i])
        except:
            msg = 'Missing reach/station '+ reach + '/'+str(station)
            print(msg)
        pyplot.title('Stage in ANUGA (dashed) and HECRAS (solid) '+reach+' reach')
        pyplot.legend()
   
    
pyplot.figure()
compare_reach('MIDDLE')
pyplot.savefig('MIDDLE_REACH.png') 
pyplot.clf()
compare_reach('RIGHT')
pyplot.savefig('RIGHT_REACH.png') 
pyplot.clf()
compare_reach('LEFT')
pyplot.savefig('LEFT_REACH.png') 

