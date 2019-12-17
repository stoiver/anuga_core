from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print "send 0"
    data = {'a': 7, 'b': 3.14}
    print data, 0
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print "recv 1"
    print data, 1
