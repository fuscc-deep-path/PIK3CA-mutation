#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `PIK3CA_mutation` package."""


import unittest
from click.testing import CliRunner

from PIK3CA_mutation import cli


class TestPathology_ai_model(unittest.TestCase):
    """Tests for `PIK3CA_mutation` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'PIK3CA_mutation.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
