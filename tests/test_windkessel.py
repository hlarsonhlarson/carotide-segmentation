import pytest

from server.windkessel import Windkessel, calibrate_two_outlet_rcr


def test_calibrated_constant_flow_preserves_target_pressure():
    flows = {"ica": 3e-6, "eca": 2e-6}
    pressures = {"ica": 11000., "eca": 10000.}
    venous = 500.
    parameters = calibrate_two_outlet_rcr(
        flows["ica"], flows["eca"], pressures["ica"], pressures["eca"],
        p_venous=venous,
    )
    for name, (rp, compliance, rd) in parameters.items():
        model = Windkessel(rp, compliance, rd, venous,
                           pressures[name] - rp * flows[name])
        for _ in range(100):
            assert model.step(.005, flows[name]) == pytest.approx(pressures[name])


def test_zero_flow_decay_matches_backward_euler_solution():
    rp, compliance, rd, dt = 1e8, 1e-9, 9e8, .005
    venous, initial = 500., 10000.
    model = Windkessel(rp, compliance, rd, venous, initial)
    for step in range(1, 201):
        expected = venous + (initial - venous) / (1 + dt / (rd * compliance)) ** step
        assert model.step(dt, 0.) == pytest.approx(expected)


def test_rc_charge_balance_with_time_varying_and_reverse_flow():
    model = Windkessel(1e8, 1e-9, 9e8, 500., 9000.)
    dt = .005
    for flow in (1e-6, 6e-6, 0., -2e-6, 4e-6):
        previous = model.P
        applied = model.step(dt, flow)
        assert model.C * (model.P - previous) / dt == pytest.approx(
            flow - (model.P - model.P_venous) / model.Rd)
        assert applied == pytest.approx(model.P + model.Rp * flow)


def test_boundary_relaxation_preserves_rc_charge_balance():
    from server.su2_pulsatile_runner import advance_outlet
    model = Windkessel(1e8, 1e-9, 9e8, 500., 9000.)
    applied, dt = 9500., .005
    for flow in (1e-6, 6e-6, 0., -2e-6, 4e-6):
        previous = model.P
        applied, clamped = advance_outlet(model, dt, flow, applied)
        assert not clamped
        assert model.C * (model.P - previous) / dt == pytest.approx(
            flow - (model.P - model.P_venous) / model.Rd)


@pytest.mark.parametrize("initial,limit", [(1e6, "P_APPLIED_MAX_PA"), (-1e6, "P_APPLIED_MIN_PA")])
def test_active_pressure_clamp_projects_and_reports_state(initial, limit):
    from server import su2_pulsatile_runner as runner
    model = Windkessel(1e8, 1e-9, 9e8, P_init=initial)
    applied, clamped = runner.advance_outlet(model, .005, 2e-6, initial)
    assert clamped
    assert applied == getattr(runner, limit)
    assert model.P + model.Rp * 2e-6 == pytest.approx(applied)
