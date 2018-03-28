from pyswmm import Simulation, Nodes, Links

from pprint import pprint
print 'Import OK'
# with Simulation('./swmm_pipe_test.inp') as sim:

def run_swmm():
    
    import pdb
    sim = Simulation('./swmm_pipe_test_2_Links_Surcharge.inp')
    print 'sim'
    pprint(dir(sim))

    node_names = ['N-1', 'N-2', 'DES-1']
    link_names = ['C-1', 'C-2']
    

    nodes = [Nodes(sim)[names] for names in node_names]
    links = [Links(sim)[names] for names in link_names]
    
    print 'node'
    pprint(dir(nodes[0]))
       
    print 'link'
    pprint(dir(links[0]))
    #sim.step_advance()
    
    sim.start()
    
    nodes[0].generated_inflow(-1)
    nodes[1].generated_inflow(1)    

    # this step_advance should be an integer multiple of the routing step
    # which is set in the ,inp file. Currently set to 10s.
    # Should be able to interrogate sim to find out what the
    # routing stepsize is. Maybe should issue a warning if
    # step_advance is set lower than the routing step size.
    # Indeed maybe step_advance should just allow advance n routing steps?

    for ind, step in enumerate(sim):
        #print(step.getCurrentSimulationTime())
        sim.step_advance(1.0)

        print 50 * "="
        
        elapsed_time = (sim.current_time - sim.start_time).total_seconds()
        
        print 'current Time', sim.current_time
        print 'elapsed time', elapsed_time
        print 'Advance seconds', sim._advance_seconds
    
        for i,j in enumerate(nodes):
            print 50*"="
            jstr = node_names[i]
            print jstr+' total_inflow', j.total_inflow
            print jstr+' total_outflow', j.total_outflow   
            print jstr+' depth' , j.depth
            print jstr+' volume' , j.volume
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
        
            
        #model_time_days = sim.__next__()
        #Evolve using either swmm_step or swmm_stride
        #model_time_days = sim._model.swmm_stride(sim._advance_seconds)
        #time = sim.evolve_step()

        #print sim
        # 'ds_xsection_area', 'flow', 'flow_limit', 'froude'
        pdb.set_trace()
        

    elapsed_time = (sim.current_time - sim.start_time).total_seconds()
    
    print 'current Time', sim.current_time
    print 'elapsed time', elapsed_time
    print 'Advance seconds', sim._advance_seconds

    for i,j in enumerate(nodes):
        print 50*"="
        jstr = 'j'+str(i)
        print jstr+' inflow', j.total_inflow
        print jstr+' outflow', j.total_outflow   
        print jstr+' depth' , j.depth
        print jstr+' volume' , j.volume
    

    for i,l in enumerate(links):
        print 50*"="
        lstr = 'l'+str(i)
        print lstr+' link flow', l1.flow
        print lstr+' Area', l1.ds_xsection_area   
        print lstr+' Froude ', l1.froude
        print lstr+' Flow limit' , l1.flow_limit
    

    
    sim.report()
    sim.close()
        

run_swmm()
