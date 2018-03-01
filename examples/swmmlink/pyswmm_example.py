from pyswmm import Simulation, Nodes, Links

from pprint import pprint
print 'Import OK'
# with Simulation('./swmm_pipe_test.inp') as sim:

def run_swmm():
    
    import pdb
    sim = Simulation('./swmm_pipe_test.inp')
    print 'sim'
    pprint(dir(sim))

    
    #pdb.set_trace()
    j1 = Nodes(sim)['N-1']
    j2 = Nodes(sim)['DES-1']
    l1 = Links(sim)['C-1']
    
    print 'For a Node....'
    print 'j1 OK'
    pprint(dir(j1))
    #pprint(j1)
    print 'j2 OK'
    pprint(dir(j2))
    print 'For a Link....'
    print 'l1 OK'
    pprint(dir(l1))
    
    #sim.step_advance()
    
    sim.start()
    
    j1.generated_inflow(9)

    # this step_advance should be an integer multiple of the routing step
    # which is set in the ,inp file. Currently set to 10s.
    # Should be able to interrogate sim to find out what the
    # routing stepsize is. Maybe should issue a warning if
    # step_advance is set lower than the routing step size.
    # Indeed maybe step_advance should just allow advance n routing steps?
    #sim.step_advance(10.0) # seconds?
    
    while (sim.evolve_step() > 0.0):
        print 50 * "="
        
        elapsed_time = (sim.current_time - sim.start_time).total_seconds()
        print 'current Time', sim.current_time
        print 'elapsed time', elapsed_time
        print 'Advance seconds', sim._advance_seconds
    
        print 'j1 inflow', j1.total_inflow
        print 'j1 outflow', j1.total_outflow   
        print 'j1 depth' , j1.depth
        print 'j1 volume' , j1.volume
        
        print 'j2 inflow', j2.total_inflow
        print 'j2 outflow', j2.total_outflow   
        print 'j2 depth' , j2.depth 
        print 'j2 volume' , j2.volume       
        
        print 'l1 link flow', l1.flow
        print 'l1 Area', l1.ds_xsection_area   
        print 'l1 Froude ', l1.froude
        print 'l1 Flow limit' , l1.flow_limit
        
        
        if elapsed_time > 25:
            j2.generated_inflow(-5)
            
        #model_time_days = sim.__next__()
        #Evolve using either swmm_step or swmm_stride
        #model_time_days = sim._model.swmm_stride(sim._advance_seconds)
        #time = sim.evolve_step()

        #print sim
        # 'ds_xsection_area', 'flow', 'flow_limit', 'froude'
        pdb.set_trace()
        

    print 'current Time', sim.current_time
    print 'end time', sim.end_time
    print 'elapsed time', (sim.current_time - sim.start_time).total_seconds()
    
    
    sim.report()
    sim.close()
        

run_swmm()
