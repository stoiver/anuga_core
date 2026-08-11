#!/usr/bin/env python
#
# Tests for the logfile tee, in particular that output written by the C
# extensions (printf to file descriptor 1) is captured as well as print().
#
# The output-capturing tests run in a subprocess on purpose: under pytest,
# sys.stdout is replaced by a capture object that never writes to file
# descriptor 1, so print() would not reach the tee and the test would be
# measuring pytest rather than ANUGA.

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import anuga.utilities.log as log


PREAMBLE = """
import ctypes
import anuga.utilities.log as log

def c_printf(text):
    libc = ctypes.CDLL(None)
    libc.printf(b'%s', text.encode('utf-8'))
    libc.fflush(None)

log.set_logfile(LOGFILE)
"""


def run_logging_script(body, logfile):
    """Run *body* in a fresh interpreter with a logfile active; return the log."""
    script = 'LOGFILE = %r\n' % logfile + PREAMBLE + textwrap.dedent(body)
    handle, path = tempfile.mkstemp(suffix='.py', prefix='anuga_log_')
    os.close(handle)
    try:
        with open(path, 'w') as fid:
            fid.write(script)
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        with open(logfile) as fid:
            return fid.read(), proc.stdout
    finally:
        os.remove(path)


class logTestCase(unittest.TestCase):

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix='.log', prefix='anuga_')
        os.close(handle)

    def tearDown(self):
        try:
            os.remove(self.path)
        except OSError:
            pass

    def test_print_goes_to_logfile_and_terminal(self):
        logged, printed = run_logging_script("""
            print('a python print')
            """, self.path)
        assert 'a python print' in logged
        assert 'a python print' in printed

    def test_c_output_goes_to_logfile(self):
        """C printf bypasses sys.stdout — the fd-level tee must still catch it."""
        logged, printed = run_logging_script("""
            c_printf('output from C\\n')
            """, self.path)
        assert 'output from C' in logged
        assert 'output from C' in printed

    def test_logfile_keeps_write_order(self):
        """print(), C output and log records must interleave in write order."""
        logged, _ = run_logging_script("""
            print('first')
            c_printf('second\\n')
            log.info('third')
            print('fourth')
            """, self.path)
        positions = [logged.index(word)
                     for word in ('first', 'second', 'third', 'fourth')]
        assert positions == sorted(positions), logged

    def test_log_records_are_not_duplicated(self):
        """The record must appear once — formatted — not also as raw text."""
        logged, _ = run_logging_script("""
            log.info('only once please')
            """, self.path)
        assert logged.count('only once please') == 1

    def test_file_only_hides_terminal_but_keeps_logfile(self):
        """file_only() must also hide C output, which it never used to."""
        logged, printed = run_logging_script("""
            with log.file_only():
                print('quiet python')
                c_printf('quiet C\\n')
            print('loud python')
            """, self.path)
        assert 'quiet python' in logged
        assert 'quiet C' in logged
        assert 'quiet python' not in printed
        assert 'quiet C' not in printed
        assert 'loud python' in printed

    def test_each_run_starts_a_fresh_logfile(self):
        """set_logfile() truncates: a run must not inherit the previous log."""
        first, _ = run_logging_script("""
            print('from the first run')
            """, self.path)
        assert 'from the first run' in first

        second, _ = run_logging_script("""
            print('from the second run')
            """, self.path)
        assert 'from the second run' in second
        assert 'from the first run' not in second

    def test_reinit_does_not_wipe_the_current_log(self):
        """Rebuilding the handlers mid-run must not truncate what was logged."""
        logged, _ = run_logging_script("""
            print('early output')
            log.close_logfile()
            log.info('later record')
            """, self.path)
        assert 'early output' in logged
        assert 'later record' in logged

    def test_close_logfile_restores_stdout(self):
        """fd 1 must be handed back, so later output is not swallowed."""
        before = os.fstat(1)
        log.set_logfile(self.path)
        log.close_logfile()
        after = os.fstat(1)
        log.log_filename = None
        log._setup = False
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)

    def test_close_logfile_is_idempotent(self):
        log.set_logfile(self.path)
        log.close_logfile()
        log.close_logfile()
        log.log_filename = None
        log._setup = False


if __name__ == '__main__':
    suite = unittest.makeSuite(logTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
