from coastseg import intersections
import numpy as np
from shapely.geometry import LineString


class TestGeometryConversion:
    """Test geometry conversion utilities."""

    def test_arr_to_linestring_basic(self):
        """Test converting coordinate array to LineString."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        result = intersections.arr_to_LineString(coords)

        assert isinstance(result, LineString)
        assert list(result.coords) == coords

    def test_arr_to_linestring_empty(self):
        """Test converting empty array to LineString."""
        coords = []
        result = intersections.arr_to_LineString(coords)

        assert isinstance(result, LineString)
        assert len(result.coords) == 0

    def test_linestring_to_arr_basic(self):
        """Test converting LineString to coordinate array."""
        coords = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        line = LineString(coords)
        result = intersections.LineString_to_arr(line)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, np.array(coords))

    def test_linestring_to_arr_single_point(self):
        """Test converting single-point LineString to array."""
        coords = [(0.0, 0.0), (0.0, 0.0)]  # LineString needs at least 2 points
        line = LineString(coords)
        result = intersections.LineString_to_arr(line)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
