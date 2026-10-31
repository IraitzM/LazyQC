"""Native thread policy for Qulacs (issue #24)."""

import pytest

from lazyqc import Pauli, Program, plan_all
from lazyqc import threads as policy_module
from lazyqc.threads import SMALL_TASK_QUBITS, WIDE_TEAM_CAP, native_thread_limit


class FakeLibrary:
    """Stands in for a threadpoolctl library controller."""

    def __init__(self, threads=8, filepath="/site-packages/qulacs.libs/libgomp.so"):
        self.num_threads = threads
        self.filepath = filepath
        self.user_api = "openmp"

    def set_num_threads(self, count):
        self.num_threads = count


@pytest.fixture
def fake(monkeypatch):
    library = FakeLibrary()
    monkeypatch.setattr(policy_module, "_qulacs_libraries", lambda: [library])
    return library


def test_auto_runs_small_tasks_on_one_thread_and_restores(fake):
    context, label = native_thread_limit("auto", [4, 10, SMALL_TASK_QUBITS - 1])
    assert "1 (auto" in label and fake.num_threads == 8  # nothing changes until entered
    with context:
        assert fake.num_threads == 1
    assert fake.num_threads == 8


def test_auto_caps_the_team_for_wide_tasks_and_never_raises_it(fake):
    fake.num_threads = 20
    context, label = native_thread_limit("auto", [4, SMALL_TASK_QUBITS])
    with context:
        assert fake.num_threads == WIDE_TEAM_CAP
    assert fake.num_threads == 20 and label.startswith("capped at")
    fake.num_threads = 4  # already below the cap: left alone
    context, label = native_thread_limit("auto", [SMALL_TASK_QUBITS])
    with context:
        assert fake.num_threads == 4
    assert label.startswith("default (team already at most")
    assert native_thread_limit("auto", [])[1] == "default"


def test_default_and_none_never_touch_the_team(fake):
    for choice in ("default", None):
        context, label = native_thread_limit(choice, [4])
        with context:
            assert fake.num_threads == 8
        assert label == "default"


def test_an_integer_fixes_the_team_size(fake):
    context, label = native_thread_limit(3, [30])
    with context:
        assert fake.num_threads == 3
    assert fake.num_threads == 8 and label == "3 (requested)"


def test_the_team_is_restored_when_execution_fails(fake):
    context, _ = native_thread_limit("auto", [4])
    with pytest.raises(RuntimeError), context:
        raise RuntimeError("backend failed")
    assert fake.num_threads == 8


@pytest.mark.parametrize("bad", ["fast", 0, -2, 1.5, True])
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(ValueError, match="native_threads"):
        native_thread_limit(bad, [4])


def test_an_uncontrollable_runtime_falls_back_to_the_default(monkeypatch):
    monkeypatch.setattr(policy_module, "_qulacs_libraries", lambda: [])
    context, label = native_thread_limit("auto", [4])
    with context:
        pass
    assert label == "default (OpenMP runtime not controllable)"


def small_plan():
    program = Program(3).h(0).cx(0, 1)
    return plan_all([program.expectation({0: "Z", 1: "Z"})])


def test_collect_reports_the_policy_and_keeps_the_values(fake):
    plan = small_plan()
    auto = plan.collect()
    assert auto.stats.native_threads.startswith("1 (auto")
    assert fake.num_threads == 8
    off = plan.collect(native_threads="default")
    assert off.stats.native_threads == "default"
    assert auto.values == off.values


def test_collect_rejects_an_invalid_policy():
    with pytest.raises(ValueError, match="native_threads"):
        small_plan().collect(native_threads="fast")


def test_a_wide_task_is_not_forced_to_one_thread(fake):
    n = SMALL_TASK_QUBITS
    program = Program(n)
    for q in range(n - 1):
        program = program.cx(q, q + 1)
    # A full-register observable keeps a task of exactly SMALL_TASK_QUBITS qubits.
    request = program.expectation(Pauli(tuple((q, "Z") for q in range(n))))
    result = plan_all([request], strategy="full").collect()
    assert result.stats.native_threads.startswith("default (team already at most")
    assert fake.num_threads == 8


def test_the_real_openmp_runtime_is_limited_and_restored():
    libraries = policy_module._qulacs_libraries()
    if not libraries:
        pytest.skip("threadpoolctl cannot control this OpenMP runtime")
    before = [lib.num_threads for lib in libraries]
    context, _ = native_thread_limit(2, [4])
    with context:
        assert all(lib.num_threads == 2 for lib in libraries)
    assert [lib.num_threads for lib in libraries] == before
