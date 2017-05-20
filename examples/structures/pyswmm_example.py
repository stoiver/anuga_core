from pyswmm import Simulation, Nodes

print 'Import OK'
with Simulation('./swmm_pipe_test.inp') as sim:
    print 'With OK'
    
    j1 = Nodes(sim)['N-1']
    print 'j1 OK', dir(j1)
    for step in sim:
        print 'step'
        print 'inflow', j1.total_inflow
        print 'outflow', j1.total_outflow        
        j1.generated_inflow(9)

