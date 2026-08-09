#!/usr/bin/env python3
"""
anuga_toml_run — run an ANUGA scenario from a TOML configuration file.

Usage (serial):
    anuga_toml_run  path/to/scenario.toml

Usage (parallel, N processes):
    mpirun -np N anuga_toml_run  path/to/scenario.toml

The working directory is changed to the directory that contains the TOML
file before the simulation starts, so all relative paths inside the TOML
are resolved relative to that directory.

A 'user_functions.py' module may be placed alongside the TOML file to
provide custom callbacks (print_velocity_statistics, print_operator_inputs).
If the file is absent those hooks are silently skipped.
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(
    description='Run an ANUGA scenario defined by a TOML configuration file.')
parser.add_argument(
    'config',
    metavar='CONFIG.toml',
    help='Path to the TOML scenario configuration file.')
parser.add_argument(
    '-n', '--dry-run', action='store_true',
    help='Do not run the simulation. Preview the scenario in the format given '
         'by --format (default: open an HTML summary in the browser).')
parser.add_argument(
    '--format', choices=('text', 'html', 'browser'), default='browser',
    help='Dry-run output. "browser" (default): write the HTML summary and open '
         'it. "html": write the HTML summary only. "text": print the '
         'highlighted, folded config to the terminal (no browser needed).')
parser.add_argument(
    '--summary-output', metavar='FILE.html', default=None,
    help='For --format html/browser: where to write the summary '
         '(default: <config>_summary.html next to the TOML).')
# --format text options
parser.add_argument(
    '--full', action='store_true',
    help='--format text: do not collapse repeated blocks; show every one.')
parser.add_argument(
    '--no-color', action='store_true',
    help='--format text: disable syntax highlighting.')
parser.add_argument(
    '--no-pager', action='store_true',
    help='--format text: write straight to stdout instead of a pager.')
parser.add_argument(
    '--threshold', type=int, default=6, metavar='N',
    help='--format text: collapse a run only when it has more than N identical '
         'blocks (default: 6).')
args = parser.parse_args()

config_path = os.path.abspath(args.config)
if not os.path.exists(config_path):
    sys.exit(f'ERROR: config file not found: {config_path}')

# Change to the scenario directory so all relative paths in the TOML resolve
# correctly.  Do this before any anuga imports that may write files.
scenario_dir = os.path.dirname(config_path)
os.chdir(scenario_dir)
config_basename = os.path.basename(config_path)

# ---------------------------------------------------------------------------
# Dry run: preview the scenario without building a mesh or evolving.  Done
# before the heavy ANUGA/mesh imports so it stays fast.
#   text    -> highlighted, folded config to the terminal
#   html    -> write the HTML summary file
#   browser -> write the HTML summary and open it
# ---------------------------------------------------------------------------
if args.dry_run:
    if args.format == 'text':
        from anuga.utilities.toml_view import render, page
        with open(config_path, encoding='utf-8') as _fh:
            _text = _fh.read()
        _color = (not args.no_color) and (sys.stdout.isatty() or not args.no_pager)
        _out = render(_text, collapse=not args.full, color=_color,
                      threshold=args.threshold)
        if args.no_pager:
            sys.stdout.write(_out)
        else:
            page(_out)
    else:
        from anuga.scenario.scenario_summary import write_scenario_summary
        out = write_scenario_summary(
            config_path, output_html=args.summary_output,
            base_dir=scenario_dir, open_browser=(args.format == 'browser'))
        print(f'Scenario summary written to: {out}')
    sys.exit(0)

# ---------------------------------------------------------------------------
# ANUGA imports (after chdir so parallel init finds the right cwd)
# ---------------------------------------------------------------------------

import anuga
from anuga.parallel import myid, numprocs, finalize, barrier
from anuga.operators.collect_max_quantities_operator import \
    Collect_max_quantities_operator

from anuga.scenario import (
    setup_boundary_conditions,
    setup_rainfall,
    setup_inlets,
    setup_bridges,
    setup_pumping_stations,
    setup_mesh,
    setup_initial_conditions,
    setup_riverwalls,
    setup_erosion,
    raster_outputs,
)
from anuga.scenario.prepare_data import PrepareData

# ---------------------------------------------------------------------------
# Optional user_functions module (lives alongside the TOML file)
# ---------------------------------------------------------------------------

sys.path.insert(0, scenario_dir)
try:
    import user_functions
    _have_user_functions = True
except ImportError:
    _have_user_functions = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

t0 = time.time()


def progress(msg):
    """Print a setup milestone to the terminal regardless of log redirection."""
    sys.__stdout__.write(msg + '\n')
    sys.__stdout__.flush()


def compute_water_balance(domain):
    """Return (v0, fs, bf, vol, imbalance) for the mass-balance identity
    ``V = V0 + BF + FS``, or None if unavailable (non-DE).

    Collective: the volume getters reduce across ranks, so this must be called
    on every rank (print on rank 0 only). v0=initial volume, fs=fractional-step
    inflow (rainfall/inlets/operators), bf=net boundary flux, vol=current volume.
    """
    stats = domain.report_water_volume_statistics(verbose=False, returnStats=True)
    if stats is None:
        return None
    vol, bf, fs = stats
    v0 = domain.volume_history[0] if getattr(domain, 'volume_history', None) else 0.0
    return (v0, fs, bf, vol, vol - v0 - bf - fs)


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

project = PrepareData(config_basename, output_log='Simulation_logfile.log')

# ---------------------------------------------------------------------------
# Build mesh and set initial conditions
# ---------------------------------------------------------------------------

progress('Building mesh')
domain = setup_mesh.setup_mesh(project)
# Phase timings recorded by setup_mesh (build vs partition/distribute).
mesh_build_time = getattr(domain, '_mesh_build_time', None)
mesh_distribute_time = getattr(domain, '_mesh_distribute_time', None)

# Propagate the scenario's coordinate reference system to the domain so it is
# written into the SWW file (zone / hemisphere / EPSG). Without this the SWW
# defaults to zone -1 / no EPSG, losing the georeferencing. project.proj4string
# is derived from projection_information (UTM zone int, "EPSG:<code>", or a
# proj4 string), so pyproj gives a single EPSG code covering all three forms.
try:
    from pyproj import CRS
    _epsg = CRS.from_proj4(project.proj4string).to_epsg()
    if _epsg is not None:
        domain.set_epsg(_epsg)
        progress('Domain CRS set to EPSG:%d' % _epsg)
    else:
        progress('Could not resolve an EPSG code from projection %r; '
                 'SWW CRS metadata may be incomplete' % project.proj4string)
except Exception as _e:
    progress('Could not set domain CRS: %s' % _e)

progress('Setting initial conditions')
setup_initial_conditions.setup_initial_conditions(domain, project)

# Riverwalls must be added AFTER any distribute step
progress('Adding riverwalls')
setup_riverwalls.setup_riverwalls(domain, project)

# ---------------------------------------------------------------------------
# Forcing terms
# ---------------------------------------------------------------------------

progress('Making rainfall')
setup_rainfall.setup_rainfall(domain, project)

progress('Making inlets')
setup_inlets.setup_inlets(domain, project)

progress('Making bridges')
setup_bridges.setup_bridges(domain, project)

progress('Making pumping stations')
setup_pumping_stations.setup_pumping_stations(domain, project)

# Erosion operators change elevation as the run proceeds. Added after the
# forcing terms and before boundary conditions, matching the ordering of the
# other operator setups.
progress('Making erosion operators')
setup_erosion.setup_erosion(domain, project)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

progress('Making boundary conditions')
setup_boundary_conditions.setup_boundary_conditions(domain, project)

# ---------------------------------------------------------------------------
# Track maximum quantities
# ---------------------------------------------------------------------------

max_quantities = Collect_max_quantities_operator(
    domain,
    update_frequency=project.max_quantity_update_frequency,
    collection_start_time=project.max_quantity_collection_start_time,
    velocity_zero_height=1.0e-03)

# ---------------------------------------------------------------------------
# Evolve
# ---------------------------------------------------------------------------

if hasattr(project, 'compute_mode'):
    domain.set_compute_mode(project.compute_mode)   # 'legacy' or 'unified'
else:
    domain.set_multiprocessor_mode(project.multiprocessor_mode)  # Excel back-compat
domain.set_omp_num_threads(project.omp_num_threads)

progress('Evolving')

barrier()
evolve_start = time.time()
for t in domain.evolve(yieldstep=project.yieldstep,
                       finaltime=project.finaltime,
                       outputstep=project.outputstep):
    if myid == 0:
        domain.print_timestepping_statistics()

    if project.report_mass_conservation_statistics:
        _wb = compute_water_balance(domain)   # collective (all ranks)
        if myid == 0 and _wb is not None:
            _v0, _fs, _bf, _vol, _resid = _wb
            _den = max(abs(_vol), abs(_v0) + abs(_fs) + abs(_bf), 1.0)
            print('    water balance: V=%.2f  FS(rain+inlet)=%.2f  BF=%.2f  '
                  'imbalance=%.3g (%.2e rel)'
                  % (_vol, _fs, _bf, _resid, _resid / _den))

    if project.report_peak_velocity_statistics and _have_user_functions:
        user_functions.print_velocity_statistics(domain, max_quantities)

    if project.report_smallest_edge_timestep_statistics:
        domain.report_cells_with_small_local_timestep()

    if project.report_operator_statistics and _have_user_functions:
        user_functions.print_operator_inputs(domain)

barrier()
evolve_time = time.time() - evolve_start

# Final water balance (DE only). Collective: compute on ALL ranks, print on
# rank 0 below. Identity: final volume = initial + boundary flux + fractional-
# step (rainfall/inlet/operator) inflow.
water_balance = None
try:
    water_balance = compute_water_balance(domain)
except Exception as _e:
    if myid == 0:
        progress(f'Water balance unavailable: {_e}')

# ---------------------------------------------------------------------------
# Phase timing + water-balance summary (rank 0). Written to the real terminal
# via progress() so it shows even when stdout is redirected to the log file.
# ---------------------------------------------------------------------------
if myid == 0:
    progress('')
    progress('Phase timings (seconds):')
    if mesh_build_time is not None:
        progress('  mesh construction : %10.2f' % mesh_build_time)
    if mesh_distribute_time is not None and numprocs > 1:
        progress('  distribute        : %10.2f' % mesh_distribute_time)
    progress('  evolve            : %10.2f' % evolve_time)
    progress('  total (wall)      : %10.2f' % (time.time() - t0))

    if water_balance is not None:
        v0, fs, bf, vol, resid = water_balance
        denom = max(abs(vol), abs(v0) + abs(fs) + abs(bf), 1.0)
        progress('')
        progress('Water balance (m^3):')
        progress('  initial volume            : %14.2f' % v0)
        progress('  rainfall + inlets (FS)    : %14.2f' % fs)
        progress('  net boundary flux (BF)    : %14.2f' % bf)
        progress('  final volume              : %14.2f' % vol)
        progress('  imbalance (V-V0-BF-FS)    : %14.2f  (%.2e relative)'
                 % (resid, resid / denom))

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

max_quantity_file_start = domain.get_datadir() + '/Max_quantities_'
max_quantities.export_max_quantities_to_csv(max_quantity_file_start)

os.chdir(project.output_dir)
if myid == 0 and numprocs > 1:
    print('Number of processors %g ' % numprocs)
    print('That took %.2f seconds' % (time.time() - t0))
    print('Communication time %.2f seconds' % domain.communication_time)
    print('Reduction Communication time %.2f seconds'
          % domain.communication_reduce_time)
    print('Broadcast time %.2f seconds'
          % domain.communication_broadcast_time)

    anuga.utilities.sww_merge.sww_merge_parallel(
        project.scenario,
        np=numprocs, verbose=True, delete_old=True)

if myid == 0:
    try:
        raster_outputs.make_me_some_tifs(
            sww_file='./' + project.scenario + '.sww',
            bounding_polygon=project.bounding_polygon,
            proj4string=project.proj4string,
            cell_size=project.output_tif_cellsize)
    except Exception as e:
        print('GeoTif creation failed: ' + str(e))
        print('You can try manually using raster_outputs.py or '
              'anuga.utilities.plot_utils.Make_Geotif')

barrier()
finalize()
