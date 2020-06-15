"""
Simple water flow example using ANUGA
Transcritical flow over a bump with a shock
"""

#------------------------------------------------------------------------------
# Import necessary modules
#------------------------------------------------------------------------------
import sys
import anuga
from anuga import Domain as Domain
from math import cos
from numpy import zeros, ones, float
from time import localtime, strftime, gmtime
from anuga import myid, finalize, distribute


#-------------------------------------------------------------------------------
# Copy scripts to time stamped output directory and capture screen
# output to file
#-------------------------------------------------------------------------------
time = strftime('%Y%m%d_%H%M%S',localtime())

#output_dir = 'transcritical_'+time
output_dir = '.'
output_file = 'transcritical'

#anuga.copy_code_files(output_dir,__file__)
#start_screen_catcher(output_dir+'_')

args = anuga.get_args()
alg = args.alg
verbose = args.verbose

#------------------------------------------------------------------------------
# Setup domain
#------------------------------------------------------------------------------
dx = 0.1
dy = dx
L = 25.
W = 3*dx

if myid == 0:
    # structured mesh
    points, vertices, boundary = anuga.rectangular_cross(int(L/dx), int(W/dy), L, W, (0.0, 0.0))
    
    #domain = anuga.Domain(points, vertices, boundary) 
    domain = Domain(points, vertices, boundary) 
    
    domain.set_name(output_file)                
    domain.set_datadir(output_dir) 
    domain.set_flow_algorithm(alg)

    #------------------------------------------------------------------------------
    # Setup initial conditions
    #------------------------------------------------------------------------------
    def elevation(x,y):
        z_b = zeros(len(x))
        for i in range(len(x)):
            if (8.0 <= x[i] <= 12.0):
                z_b[i] = 0.2 - 0.05*(x[i]-10.0)**2.0
            else:
                z_b[i] = 0.0
        return z_b
    domain.set_quantity('elevation',elevation)
    domain.set_quantity('friction', 0.0)
    
    
    def stage(x,y):
        return 2.0*ones(len(x))
    domain.set_quantity('stage', stage)
else:
    domain = None

#-----------------------------------------------------------------------------
# Create Parallel Domain
#------------------------------------------------------------------------------
domain = distribute(domain)
  
#-----------------------------------------------------------------------------
# Setup boundary conditions
#------------------------------------------------------------------------------
from math import sin, pi, exp
Br = anuga.Reflective_boundary(domain)      # Solid reflective wall
Bt = anuga.Transmissive_boundary(domain)    # Continue all values on boundary 
BdL = anuga.Dirichlet_boundary([0.41373588752426715, 0.18, 0.]) # Constant boundary values
BdR = anuga.Dirichlet_boundary([0.33, 0.18, 0.]) # Constant boundary values

# Associate boundary tags with boundary objects
domain.set_boundary({'left': BdL, 'right': BdR, 'top': Br, 'bottom': Br})


#------------------------------------------------------------------------------
# Produce a documentation of parameters
#------------------------------------------------------------------------------
if myid == 0:
    parameter_file=open('parameters.tex', 'w')
    parameter_file.write('\\begin{verbatim}\n')
    from pprint import pprint
    pprint(domain.get_algorithm_parameters(),parameter_file,indent=4)
    parameter_file.write('\\end{verbatim}\n')
    parameter_file.close()

#------------------------------------------------------------------------------
# Evolve system through time
#------------------------------------------------------------------------------
for t in domain.evolve(yieldstep = 1.0, finaltime = 100.):
    #print(domain.timestepping_statistics(track_speeds=True))
    if myid == 0 and verbose: print(domain.timestepping_statistics())
    #vis.update()

domain.sww_merge(delete_old=True)

finalize()

