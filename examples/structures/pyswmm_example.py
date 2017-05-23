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
    print 'For a Link....'
    print 'l1 OK'
    pprint(dir(l1))
    
    sim.start()
    sim.step_advance(30.0) # seconds?
    
    while (True):
        print 50 * "="
        print 'current Time', sim.current_time
        print 'j1 inflow', j1.total_inflow
        print 'j1 outflow', j1.total_outflow        
        j1.generated_inflow(9)
        print 'j1 depth' , j1.depth
        print 'l1 link flow', l1.flow
        print 'l1 Area', l1.ds_xsection_area   
        print 'l1 Froude ', l1.froude
        print 'l1 Flow limit' , l1.flow_limit
        model_time_days = sim._model.swmm_stride(sim._advance_seconds)
        print 'Advance seconds', sim._advance_seconds
        print 'model time (s)', model_time_days*3600*24
        print sim.percent_complete
        #print sim
        # 'ds_xsection_area', 'flow', 'flow_limit', 'froude'
        pdb.set_trace()
        

run_swmm()
