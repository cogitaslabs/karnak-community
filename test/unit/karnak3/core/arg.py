import unittest
import argparse

import karnak3.core.arg as kca


class TestArg(unittest.TestCase):
    def test_multiple_args(self):
        parser = argparse.ArgumentParser()
        kca.add_core_params(parser)
        kca.add_window_params(parser)
        self.assertTrue(kca.test_argument_added(parser, '--ws'))
        self.assertFalse(kca.test_argument_added(parser, '--xpto'))
        self.assertFalse(kca.test_argument_added(parser, '--xpto'))
        parser.add_argument('--xpto')
