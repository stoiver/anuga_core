#! /usr/bin/python
__author__="stephen"
__date__ ="$20/08/2012 11:20:00 PM$"



def run_script(script, args=None, np=1, alg=None, verbose=False, debug=False, allow_parallel=True):
    #from anuga.validation_utilities.fabricate import run


    if args is None:
        if alg is None:
            from anuga.validation_utilities.parameters import alg
    else:
        alg = args.alg
        np = args.np
        verbose = args.verbose
        debug = getattr(args, 'debug', False)


    #print args
    args_dict = vars(args)
    #print args_dict
    #print zip(args_dict.keys(), args_dict.values())


    # Forward the standard flags to the child process
    flags = ''
    if verbose:
        flags += ' -v'
    if debug:
        flags += ' -d'


    import subprocess
    import os
    try:
        if np>1 and allow_parallel:
            cmd = 'mpiexec -np %s python %s -alg %s%s' % (str(np), script, str(alg), flags)
        else:
            cmd = 'python %s -alg %s%s' % (script, str(alg), flags)

        if verbose:
            print(50*'=')
            print('Run '+cmd)
            print(50*'=')

        #os.system(cmd)
        res = subprocess.call([cmd], shell=True)

        return res

    except Exception:
        return 1







