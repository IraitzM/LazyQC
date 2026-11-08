"""Measured placement crossover between the default and a wide backend (issue #3)."""

from types import SimpleNamespace

import pytest

from lazyqc import calibrate_crossover
from lazyqc.backends.qulacs import QulacsBackend


class Clock:
    """A fake clock that backends advance, so no test depends on wall time."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class Fake:
    """A backend whose evolution costs `cost(width, gates)` clock units."""

    def __init__(self, clock, cost):
        self.clock, self.cost, self.counters = clock, cost, None

    def lower(self, task):
        return SimpleNamespace(width=task.width, gates=len(task.operations), observables=(0,))

    def evolve(self, lowered):
        self.clock.now += self.cost(lowered.width, lowered.gates)
        return lowered

    def expectations(self, state, observables):
        return (0.0,)


def measure(narrow_cost, wide_cost, **options):
    clock = Clock()
    return calibrate_crossover(
        Fake(clock, wide_cost),
        backend=Fake(clock, narrow_cost),
        clock=clock,
        native_threads=None,
        **options,
    )


def test_the_crossover_is_the_first_width_the_wide_backend_wins():
    # 10 * 2^w overtakes 4000 + w at 9 qubits.
    width, report = measure(lambda w, g: 10 * 2**w, lambda w, g: 4000 + w)
    assert width == 9
    assert [r[0] for r in report] == list(range(6, 11))  # stops after two confirming wins
    assert report[0][1:] == (10 * 2**6, 4006)


def test_a_lone_early_win_does_not_count():
    wide = lambda w, g: 1 if w == 3 else 10**9  # noqa: E731
    narrow = lambda w, g: 10 * 2**w  # noqa: E731
    width, report = measure(narrow, wide, widths=range(1, 8))
    assert width is None
    assert [r[0] for r in report] == list(range(1, 8))


def test_never_winning_returns_none_with_the_full_report():
    width, report = measure(lambda w, g: 1, lambda w, g: 2, widths=(4, 8, 12))
    assert width is None and [r[0] for r in report] == [4, 8, 12]


def test_the_crossover_moves_with_the_gate_count():
    # The wide device pays per gate, so tasks with more gates cross later.
    narrow = lambda w, g: 3 * 2**w  # noqa: E731
    wide = lambda w, g: 500 + 200 * g  # noqa: E731
    few, _ = measure(narrow, wide, gates=5)
    many, _ = measure(narrow, wide, gates=50)
    assert few == 9 and many == 12


def test_confirm_widens_the_required_run():
    narrow = lambda w, g: 10 * 2**w  # noqa: E731
    wide = lambda w, g: 4000 + w  # noqa: E731
    width, report = measure(narrow, wide, confirm=3)
    assert width == 9 and [r[0] for r in report][-1] == 11


def test_the_result_is_deterministic_and_takes_the_best_of_the_repeats():
    values = [measure(lambda w, g: 10 * 2**w, lambda w, g: 4000 + w, repeats=r) for r in (1, 3)]
    assert values[0][0] == values[1][0] == 9


@pytest.mark.parametrize(
    "options",
    [
        {"widths": ()},
        {"widths": (5, 3)},
        {"widths": (3, 3)},
        {"widths": (0, 2)},
        {"gates": 0},
        {"repeats": 0},
        {"confirm": 0},
    ],
)
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        measure(lambda w, g: 1, lambda w, g: 1, **options)


def test_backends_must_follow_the_protocol():
    with pytest.raises(ValueError, match="wide_backend"):
        calibrate_crossover(object())


def test_runs_on_real_backends_and_reports_every_width():
    width, report = calibrate_crossover(
        QulacsBackend(), widths=(1, 2, 3), gates=4, repeats=1, confirm=1
    )
    assert width in (None, 1, 2, 3)
    assert [r[0] for r in report][0] == 1 and all(r[1] > 0 and r[2] > 0 for r in report)
