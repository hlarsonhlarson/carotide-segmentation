"""3-element (RCR) Windkessel outlet boundary condition.

Ported from the validated standalone pipeline
(~/Desktop/SU2 Pipeline Code/shared/windkessel.py), with one change:
calibrate_two_outlet_rcr here takes a *separate* target mean pressure per
outlet instead of one shared mean_pressure — the app's own Boundary
Conditions screen lets the user set ICA/ECA outlet pressure independently
(the two outlets' own downstream vascular beds don't necessarily sit at
the same operating pressure), so the calibration has to honor that
instead of assuming they're equal.

Circuit: proximal resistance Rp in series with a parallel {compliance C,
distal resistance Rd} pair, referenced to a constant distal/venous
pressure P_venous (~0, central venous pressure is small compared to
arterial pressures and mostly just sets the operating point).

    P(t) = flow-dependent pressure just inside the parallel RC block.
    Governing ODE (from KCL at the P node):
        C * dP/dt = Q_in(t) - (P - P_venous) / Rd
    Applied outlet pressure (what SU2's MARKER_OUTLET pressure BC gets):
        P_outlet(t) = P(t) + Rp * Q_in(t)

Integrated with implicit (backward) Euler for stability at the CFD time
step, since Rd*C time constants are ~1s while CFD dt is ~1e-3 s.
"""


class Windkessel:
    def __init__(self, Rp, C, Rd, P_venous=0.0, P_init=None):
        self.Rp = Rp
        self.C = C
        self.Rd = Rd
        self.P_venous = P_venous
        self.P = P_init if P_init is not None else P_venous

    def step(self, dt, Q_in):
        """Advance the internal RC state by dt given the flow rate Q_in
        [m^3/s] leaving through this outlet, backward-Euler:
            C*(P_new - P_old)/dt = Q_in - (P_new - P_venous)/Rd
            => P_new*(C/dt + 1/Rd) = C/dt*P_old + Q_in + P_venous/Rd
        Returns the outlet pressure to apply at the SU2 marker:
        P_outlet = P_new + Rp*Q_in."""
        a = self.C / dt + 1.0 / self.Rd
        b = self.C / dt * self.P + Q_in + self.P_venous / self.Rd
        self.P = b / a
        return self.P + self.Rp * Q_in


def _one(mean_flow, mean_pressure, p_venous, compliance, rp_fraction):
    R_total = (mean_pressure - p_venous) / mean_flow
    Rp = rp_fraction * R_total
    Rd = R_total - Rp
    return Rp, compliance, Rd


def calibrate_two_outlet_rcr(mean_flow_ica, mean_flow_eca, ica_mean_pressure,
                              eca_mean_pressure, p_venous=0.0,
                              total_compliance=None, rp_fraction=0.1):
    """Pick Rp/C/Rd for the ICA and ECA outlets so that, at the cycle-mean
    operating point, each outlet carries its own target mean flow at its
    own target mean pressure (total resistance per outlet =
    (mean_pressure - p_venous) / mean_flow, split Rp:Rd by rp_fraction —
    Rp mimics the outlet's own local arterial resistance/wave impedance,
    Rd the downstream bed).

    Small fixed per-outlet compliances (not total_compliance-derived) by
    default — CONFIRMED in the original pipeline that a larger,
    physiologically-realistic compliance gives an RC time constant
    (Rd*C) of several seconds, which needs many more cardiac cycles to
    reach periodic steady state than is affordable; these smaller values
    trade some realism in the pulsatility *shape* of outlet pressure/flow
    for reaching periodicity in ~1 cycle, without changing the
    Rp+Rd values that set the correct *mean* flow split."""
    if total_compliance is not None:
        total_flow = mean_flow_ica + mean_flow_eca
        c_ica = total_compliance * mean_flow_ica / total_flow
        c_eca = total_compliance * mean_flow_eca / total_flow
    else:
        c_ica = 1.2e-9
        c_eca = 0.6e-9

    return {
        "ica": _one(mean_flow_ica, ica_mean_pressure, p_venous, c_ica, rp_fraction),
        "eca": _one(mean_flow_eca, eca_mean_pressure, p_venous, c_eca, rp_fraction),
    }
