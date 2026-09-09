import pytest
def test_three_measurements(call):
    assert call("basics","three_measurements") == pytest.approx([12,4])
