import numpy as np
import pytest

from equity_derivatives import ArithmeticAsian, Barrier, European, FixedStrikeLookback


@pytest.fixture
def paths():
    # Three paths: touch up, touch down, never touch either.
    return np.array([[100., 100., 100.], [120., 80., 103.], [105., 90., 110.]])


def test_vanilla(paths):
    np.testing.assert_array_equal(European(100).evaluate(paths), [5, 0, 10])
    np.testing.assert_array_equal(European(100, False).evaluate(paths), [0, 10, 0])


def test_arithmetic_asian_observation_convention(paths):
    np.testing.assert_allclose(ArithmeticAsian(100).evaluate(paths), [12.5, 0, 6.5])
    np.testing.assert_allclose(ArithmeticAsian(100, include_initial=True).evaluate(paths),
                               [25./3, 0, 13./3])


def test_fixed_strike_lookback(paths):
    np.testing.assert_allclose(FixedStrikeLookback(100).evaluate(paths), [20, 0, 10])
    np.testing.assert_allclose(FixedStrikeLookback(100, False).evaluate(paths), [0, 20, 0])


@pytest.mark.parametrize("direction,h", [("up", 115), ("down", 85)])
def test_barrier_knock_in_plus_out_equals_vanilla_pathwise(paths, direction, h):
    x = Barrier(100, h, direction, "out").evaluate(paths)
    y = Barrier(100, h, direction, "in").evaluate(paths)
    np.testing.assert_array_equal(x+y, European(100).evaluate(paths))


def test_barrier_hit_at_initial_time_and_equal_touch():
    paths = np.array([[125., 125.], [110., 140.], [130., 90.]])
    np.testing.assert_array_equal(Barrier(100, 125, "up", "out").evaluate(paths), [0,0])


def test_barrier_monitoring_finer_cannot_create_out_payoff():
    fine = np.array([[100,100], [130,95], [103,104], [110,118], [111,109]], float)
    coarse = fine[::2]
    out = Barrier(100, 125)
    assert np.all(out.evaluate(fine) <= out.evaluate(coarse))


def test_invalid_payoff_inputs(paths):
    with pytest.raises(ValueError):
        Barrier(100, 125, "sideways")
    with pytest.raises(ValueError):
        ArithmeticAsian(-1)
    with pytest.raises(ValueError):
        European(100).evaluate(paths[:1])
