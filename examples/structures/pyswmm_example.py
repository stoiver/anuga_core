from pyswmm import Simulation, Nodes, Links

print 'Import OK'
with Simulation('./swmm_pipe_test.inp') as sim:
    print 'With OK'
    
    j1 = Nodes(sim)['N-1']
    j2 = Nodes(sim)['DES-1']
    l1 = Links(sim)['C-1']
    
    print 'For a Node....'
    print 'j1 OK', dir(j1)
    print 'For a Link....'
    print 'l1 OK', dir(l1)
    for step in sim:
        print 'step'
        print 'inflow', j1.total_inflow
        print 'outflow', j1.total_outflow        
        j1.generated_inflow(9)
        print 'depth' ,j1.depth
        print 'link flow', l1.flow
        print 'Area',l1.ds_xsection_area   
        #'ds_xsection_area', 'flow', 'flow_limit', 'froude'
        raw_input('check')

