### ubuntu 16.04LTS
import anuga, anuga.parallel, numpy, time, os, glob
from anuga.operators.rate_operators import Polygonal_rate_operator
from anuga import file_function, Polygon_function, read_polygon, create_mesh_from_regions, Domain, Inlet_operator
import anuga.utilities.spatialInputUtil as su

from anuga import distribute, myid, numprocs, finalize, barrier
from anuga.parallel.parallel_operator_factory import Inlet_operator, Boyd_box_operator, Boyd_pipe_operator
from anuga import Rate_operator
#-----------------------------------------------------------------------
# FILENAMES & DOMAIN INFORMATION
#-----------------------------------------------------------------------

basename = 'dem'
outname = 'simulation'
meshname = 'dem.tsh'

#top right coordinates
W=304880
N=6187890

#bottom left coordinates
E=305100
S=6187710

starting_tide = 0                   # this is applied to the domain boundary condition, takes time to run up
base_stage = 0                      # this is applied to the domain boundary condition at t=0, no runup time
base_friction = 0.035               # manning's n everywhere except inside your polies
maximum_triangle_area = 1           # size of mesh everywhere except inside your polies 
minimum_storable_height = 0.01      # minimum depth to store during run
yieldstep = 60                      # timestep at which model run is saved
finaltime = 7200                    # time at end of model run

#-----------------------------------------------------------------------
# SETUP DOMAIN ONLY ON PROCESSOR 0
#-----------------------------------------------------------------------

if anuga.parallel.myid == 0:
    #-------------------------------------------------------------------
    # CREATING MESH
    #-------------------------------------------------------------------
    
    bounding_polygon = [[W, S], [E, S], [E, N], [W, N]]
    domain = anuga.create_domain_from_regions(bounding_polygon,
        boundary_tags={'south': [0], 'east': [1], 'north': [2], 'west': [3]},
        maximum_triangle_area=maximum_triangle_area,
        mesh_filename=meshname,
        use_cache=False, # if you want to change the mesh size, you have to set this to FALSE so that the model can recalculate the new mesh. Set to TRUE if u want to use the mesh already created
        verbose=True)
        
    domain.set_minimum_storable_height(minimum_storable_height) 
    domain.set_name(outname)
    domain.set_quantity('elevation', filename=basename+'.csv', use_cache=True, verbose=True, alpha=0.99)
    print domain.statistics() 
     
    domain.set_quantity('friction', base_friction)
    domain.set_quantity('stage', base_stage)

else:
    domain = None
if anuga.parallel.myid == 0 and True: print 'DISTRIBUTING DOMAIN'
domain = anuga.parallel.distribute(domain)
if anuga.parallel.myid == 0 and True: print 'CREATING INLETS'  
 
#-----------------------------------------------------------------------
# APPLY RAINFALL
#-----------------------------------------------------------------------
print 
print 'Applying rainfall via polygons over the domain'
Gaugefile = '10y25m.csv'
Rainfile = '10y25m.tms'
op1 = anuga.Rate_operator(domain, rate=anuga.file_function(Rainfile, quantities='rate'), factor=1.0e-3 , polygon=anuga.read_polygon(Gaugefile), default_rate = 0.0)

#-----------------------------------------------------------------------
# SETUP BOUNDARY CONDITIONS
#-----------------------------------------------------------------------
print 
print 'Available boundary tags', domain.get_boundary_tags()
    
Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([starting_tide,0,0])

domain.set_boundary({'interior': Br, 'exterior': Bd, 'west': Bd, 'south': Bd, 'north': Bd, 'east': Bd})
    
#-----------------------------------------------------------------------
# EVOLVE SYSTEM THROUGH TIME
#-----------------------------------------------------------------------
if anuga.parallel.myid == 0 and True: print 'EVOLVE'   
t0 = time.time()  
for t in domain.evolve(yieldstep = yieldstep, finaltime =finaltime):   
    
    """	
	#This does not work, but i assume this is where you would place
	#send infomrtion backwards and forwards between anuga 2d and 
	# SWMM model. We dont want to do these calculations at every
	#anuga internal time step, rather at each anuga yield step???
	
	#Imported swmm pipe network model here?????????
    from swmm import * # Imported SWMM module
    
    # ***********************************************************************
    #  Declaration of simulation files and variables
    # ***********************************************************************
    
    inp    = 'swmm_pipe_network.inp'  # Input pipe network filename
    flow   = []
    vol    = []
    time   = []
    
    # ***********************************************************************
    #  Initializing SWMM
    # ***********************************************************************
    
    swmm.initialize(inp)  # Step 1
    
    # ***********************************************************************
    #  Step Running
    # ***********************************************************************
    
    # Main loop: finished when the simulation time is over.
    while( not swmm.is_over() ): 
    
    	# ----------------- Run step and retrieve simulation time -----------
    	
    	time.append( swmm.get_time() )
    	swmm.run_step()  # Step 2
    	
    	# --------- Retrieve & modify information during simulation ---------
    	# Retrieve information about flow in C-5
    	f = swmm.get('C-5', swmm.FLOW, swmm.SI)   
    	# Stores the information in the flow vector
    	flow.append(f)					 
    	# Retrieve information about volume in V-1
    	v = swmm.get('V-1', swmm.VOLUME, swmm.SI) 
    	# Stores the information in the volume vector
    	vol.append(v)					 
    
    # ************************************************************************
    #  End of simulation
    # ************************************************************************
    
    errors = swmm.finish() # Step 3
    """    	
    		      
    if anuga.parallel.myid == 0:
        domain.write_time()
if anuga.parallel.myid == 0:
    print 'Number of processors %g ' %anuga.parallel.numprocs
    print 'That took %.2f seconds' %(time.time()-t0)
    print 'Communication time %.2f seconds'%domain.communication_time
    print 'Reduction Communication time %.2f seconds'%domain.communication_reduce_time
    print 'Broadcast time %.2f seconds'%domain.communication_broadcast_time
domain.sww_merge(delete_old=True)
anuga.parallel.finalize()
