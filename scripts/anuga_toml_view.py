#!/usr/bin/env python3
"""
anuga_toml_view — view an ANUGA TOML scenario file in the terminal.

Syntax-highlights the config and, by default, collapses long runs of identical
array-of-table blocks (the many one-per-file friction / culvert / rainfall
entries) down to the first block plus a "N more" marker, so the structure of a
scenario reads at a glance.

    anuga_toml_view towradgi.toml            # highlighted, collapsed, paged
    anuga_toml_view towradgi.toml --full     # every block, nothing collapsed
    anuga_toml_view towradgi.toml --no-color # plain text
    anuga_toml_view towradgi.toml | less -R  # explicit pager

For a rendered, browsable summary (mesh, friction breakdown, rainfall
hyetograph, inlet plot) rather than the raw text, use instead:

    anuga_toml_run <config>.toml --dry-run
"""

import argparse
import os
import sys


def _page(text):
    """Send *text* to a pager when stdout is an interactive terminal.

    Honours $PAGER (default 'less -R' so ANSI colour survives); prints plainly
    when not a tty or if the pager cannot be launched.
    """
    if not sys.stdout.isatty():
        sys.stdout.write(text)
        return
    pager = os.environ.get('PAGER', 'less -R')
    try:
        import subprocess
        proc = subprocess.Popen(pager, shell=True, stdin=subprocess.PIPE)
        proc.communicate(text.encode('utf-8', 'replace'))
    except (OSError, BrokenPipeError, KeyboardInterrupt):
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='anuga_toml_view',
        description='View an ANUGA TOML scenario file, highlighted and with '
                    'long repeated blocks collapsed.')
    parser.add_argument('config', metavar='CONFIG.toml',
                        help='Path to the TOML scenario file.')
    parser.add_argument('-f', '--full', action='store_true',
                        help='Do not collapse repeated blocks; show every one.')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable syntax highlighting.')
    parser.add_argument('--threshold', type=int, default=6, metavar='N',
                        help='Collapse a run only when it has more than N '
                             'identical blocks (default: 6).')
    parser.add_argument('--no-pager', action='store_true',
                        help='Write straight to stdout instead of a pager.')
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        sys.exit(f'ERROR: file not found: {args.config}')

    with open(args.config, encoding='utf-8') as fh:
        text = fh.read()

    from anuga.utilities.toml_view import render
    # Colour only when the destination is a terminal (unless forced off).
    color = (not args.no_color) and (sys.stdout.isatty() or not args.no_pager)
    out = render(text, collapse=not args.full, color=color,
                 threshold=args.threshold)

    if args.no_pager:
        sys.stdout.write(out)
    else:
        _page(out)


if __name__ == '__main__':
    main()
