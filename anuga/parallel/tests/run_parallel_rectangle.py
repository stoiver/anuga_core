import os
        
#------------------------------------------
# anuga imports
#------------------------------------------
import anuga 

from anuga.utilities.system_tools import get_pathname_from_package

from anuga import Domain
from anuga import Reflective_boundary
from anuga import Dirichlet_boundary
from anuga import Time_boundary
from anuga import Transmissive_boundary

from anuga import rectangular_cross
from anuga import create_domain_from_file

from anuga import distribute, myid, numprocs, finalize

from anuga.parallel.parallel_shallow_water import Parallel_domain
from anuga.parallel.parallel_meshes import parallel_rectangle


#----------------------------------
# set up MPI to abort on error
#----------------------------------
from anuga.utilities.parallel_abstraction import global_except_hook
import sys
sys.excepthook = global_except_hook

# ----------------
# Setup parameters
# ----------------

verbose = True

# ----------------
# Setup procedures
# ----------------
class Set_Stage(object):
    """Set an initial condition with constant water height, for x<x0
    """

    def __init__(self, x0=5.0, x1=10.0, h=1.0):
        self.x0 = x0
        self.x1 = x1
        self.h = h

    def __call__(self, x, y):
        return self.h * ((x > self.x0) & (x < self.x1))


points, elements, boundary, full_send_dict, ghost_recv_dict = parallel_rectangle(100, 200, len1_g=10.0, len2_g=20.0, origin_g = (0.0, 0.0))

domain = Parallel_domain(points,
                         elements,
                         boundary,
                         full_send_dict=full_send_dict,
                         ghost_recv_dict=ghost_recv_dict)

print(f"Proc {myid}: # full tris {domain.number_of_full_triangles}, # # tris {domain.number_of_triangles}")


domain.set_quantity('stage', Set_Stage())

if numprocs > 1:
    if myid == 0 and verbose: print('PARALLEL EVOLVE')
    domain.set_name('rectangle_parallel')        
else:
    if verbose: print('SEQUENTIAL EVOLVE')
    domain.set_name('rectangle_sequential')        




# --------------------------------------------------------------
# Setup boundary conditions
# This must currently happen *after* domain has been distributed
# --------------------------------------------------------------
Br = Reflective_boundary(domain)      # Solid reflective wall - no movement

domain.set_boundary({'bottom': Br, 'left': Br, 'top': Br, 'right': Br, 'ghost': None})

domain.set_store(False)

# ---------
# Evolution
# ---------

for t in domain.evolve(yieldstep=1.0, finaltime=2.0):
    if myid == 0 and verbose: domain.write_time()

# ------------------------------------
# Wrap up parallel matters if required
# ------------------------------------
domain.sww_merge(delete_old=True)
finalize()



