#########################################################
#
#  Example of running a simple parallel model
#
#  Need mpi setup for your machine 
#
#  To run in parallel on 4 processes, use the following
#
#  mpiexec -np 4 python -u run_parallel_sw_rectangular_cross.py
#
#
#  Note the use of "if myid == 0" to restrict some calculations 
#  to just one processor, in particular the creation of a 
#  full domain on processor 0 which is then distributed to the
#  processors. 
#
#  Authors: 
#  Linda Stals, Steve Roberts and Matthew Hardy - June 2005
#  Steve Roberts - 2018
#
#
#
#########################################################

import time
import sys
import math
from xml import dom
import anuga


#----------------------------
# Sequential interface
#---------------------------
from anuga import Transmissive_boundary, Reflective_boundary
from anuga import rectangular_cross_domain
from anuga import Set_stage



t0 = time.time()

#----------------------------
# simulation parameters with some 
# defaults
#----------------------------
sqrtN = 100
length = 2.0
width = 2.0

yieldstep = 0.005
finaltime = 0.015

fixed_flux_timestep = 0.0

import argparse
parser = argparse.ArgumentParser(description='Rectangular')

parser.add_argument('-ft', '--finaltime', type=float, default=finaltime,
                    help='finaltime')
parser.add_argument('-ys', '--yieldstep', type=float, default=yieldstep,
                    help='yieldstep')
parser.add_argument('-sn', '--sqrtN', type=int, default=sqrtN,
                    help='Size of grid: 500 -> 1_000_000 triangles')
parser.add_argument('-gl', '--ghost_layer', type=int, default=2,
                    help='Size of ghost layer')
parser.add_argument('-np', '--numprocs', type=int, default=10,
                    help='Distibute into np sub-domains')


parser.add_argument('-fdt', '--fixed_dt', type=float, default=fixed_flux_timestep,
                    help='Set a fixed flux timestep')
parser.add_argument('-ta', '--test_allreduce', action='store_true',
                    help='run fixed timestep with dummy allreduce')

parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbosity')

parser.add_argument('-ve', '--evolve_verbose', action='store_true', help='turn on evolve verbosity')

parser.add_argument('-sww', '--store_sww', action='store_true', help='turn on storing sww file')

args = parser.parse_args()

print(args)

sqrtN = args.sqrtN
yieldstep = args.yieldstep
finaltime = args.finaltime
verbose = args.verbose
evolve_verbose = args.evolve_verbose
fixed_flux_timestep = args.fixed_dt
test_allreduce = args.test_allreduce
ghost_layer = args.ghost_layer
store_sww = args.store_sww

ncpus = args.numprocs

dist_params = {}
dist_params['ghost_layer_width'] = ghost_layer

if fixed_flux_timestep == 0.0:
    fixed_flux_timestep = None

#print('fixed_flux_timestep ',fixed_flux_timestep)

domain_name = f'rect_gl_{ghost_layer}_sqrtn_{sqrtN}_ncpus_{ncpus}'
partition_dir = 'Partitions'


#--------------------------------------------------------------------------
# Setup Domain only on a single processor
#--------------------------------------------------------------------------

domain = rectangular_cross_domain(sqrtN, sqrtN,
                                    len1=length, len2=width, 
                                    origin=(-length/2, -width/2), 
                                    verbose=verbose)


domain.set_store(store_sww)
domain.set_quantity('elevation', lambda x,y : -1.0-x )
domain.set_quantity('stage', 1.0)
domain.set_flow_algorithm('DE0')
domain.set_name(domain_name)

if verbose: domain.print_statistics()


t1 = time.time()

creation_time = t1-t0

print ('Creation of sequential domain: Time =',t1-t0)
print ('Creation of sequential domain: Number of Triangles =',domain.number_of_global_triangles)

 
print ('DISTRIBUTING DOMAIN')
sys.stdout.flush()
    
#-------------------------------------------------------------------------
# Distribute domain
#-------------------------------------------------------------------------

t2 = time.time()

anuga.sequential_distribute_dump(domain,numprocs=ncpus, verbose=verbose, partition_dir=partition_dir)

t3 = time.time()

distribute_time = t3-t2
print ('Dump Domain: Time ',distribute_time)
