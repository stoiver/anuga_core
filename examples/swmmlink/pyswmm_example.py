from pyswmm import Simulation, Nodes, Links

import numpy as np

from pprint import pprint
print 'Import OK'

def run_swmm():
    
    import pdb
    sim = Simulation('./swmm_pipe_test.inp')
    print 'sim'
    #pprint(dir(sim))

    node_names = ['Inlet', 'Outlet']
    link_names = ['Culvert']
    
    nodes = [Nodes(sim)[names] for names in node_names]
    links = [Links(sim)[names] for names in link_names]
    
    print 'node'
    #pprint(dir(nodes[0]))
       
    print 'link'
    #pprint(dir(links[0]))
    #sim.step_advance()



    #=======================================
    # setup all the nodes before starting
    #=======================================
    

    nodes[0].nodeid
    # type, area, length, orifice_coeff, free_weir_coeff, submerged_weir_coeff
    openning0 = nodes[0].create_opening(4, 1.0, 1.0, 0.6, 1.6, 1.0)

    print "n0 Is coupled? ", nodes[0].is_coupled
    print "n1 Is coupled? ", nodes[1].is_coupled



    #=======================================
    # Start the simulation
    #=======================================   
    sim.start()

    nodes[0].overland_depth = 1.0
    nodes[0].coupling_area = 1.0

    # this step_advance should be an integer multiple of the routing step
    # which is set in the ,inp file. Currently set to 1s.
    # Should be able to interrogate sim to find out what the
    # routing stepsize is. Maybe should issue a warning if
    # step_advance is set lower than the routing step size.
    # Indeed maybe step_advance should just allow advance n routing steps?
    #sim.step_advance(1.0) # seconds?
    
    bucket = np.array([1.0, 0.0])
   
    for ind, step in enumerate(sim):
        sim.step_advance(1.0)

        print 70 * "="
        
        elapsed_time = (sim.current_time - sim.start_time).total_seconds()
        
        print 'current Time', sim.current_time
        print 'elapsed time', elapsed_time
        print 'Advance seconds', sim._advance_seconds
    
        pprint(openning0.flow)
        
        for i,j in enumerate(nodes):
            print 50*"="
            jstr = node_names[i]
            print jstr+' total_inflow', j.total_inflow
            print jstr+' total_outflow', j.total_outflow
            print jstr+' coupling_inflow', j.coupling_inflow
            
            print jstr+' coupling_area', j.coupling_area
            print jstr+' overland_depth', j.overland_depth
            print jstr+' number of openings', j.number_of_openings
            print jstr+' depth' , j.depth
            print jstr+' volume' , j.volume
            print jstr+' surcharge depth', j.surcharge_depth
            print jstr+' flooding' , j.flooding
            print jstr+' lateral_inflow' , j.lateral_inflow
            
            
        

        for i,l in enumerate(links):
            print 50*"="
            lstr = link_names[i]
            print lstr+' link flow', l.flow
            
            print lstr+' Area', l.ds_xsection_area   
            print lstr+' Froude ', l.froude
            print lstr+' Depth ', l.depth
            print lstr+' Flow limit' , l.flow_limit
            print lstr+' volume' , l.volume
        


        #pdb.set_trace()
      
    print 70 * "="  

    print 'current Time', sim.current_time
    print 'end time', sim.end_time
    print 'elapsed time', (sim.current_time - sim.start_time).total_seconds()
    
    
    sim.report()
    sim.close()
    
    print ''
        

run_swmm()
#pdb.set_trace()

