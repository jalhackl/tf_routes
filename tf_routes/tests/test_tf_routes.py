"""
Unit and regression test for the tf_routes package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import tf_routes


def test_tf_routes_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tf_routes" in sys.modules
