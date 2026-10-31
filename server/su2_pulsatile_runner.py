#!/usr/bin/env python3
"""Pulsatile carotid-bifurcation CFD driver for SU2 (pysu2) — this app's
own subprocess entry point for the actual CFD solve, run via:

    mpirun -n <N> python3 server/su2_pulsatile_runner.py run.cfg run_params.json

Adapted from the validated standalone pipeline's pulsatile_run_generic.py
(~/Desktop/SU2 Pipeline Code/shared/pulsatile_run_generic.py — same
physics/numerics, same MPI halo-safe flux integration, same under-relaxed
Windkessel coupling), with two changes: (1) each outlet's Windkessel is
calibrated against its *own* target mean pressure (ica_map_pa/eca_map_pa
in run_params.json) instead of one shared mean_arterial_pressure, matching
this app's own independently-editable ICA/ECA outlet pressure controls;
(2) progress is reported as a single-line JSON blob prefixed with
SU2PROGRESS/SU2DONE/SU2ERROR on stdout every step, rather than a human-
readable summary every 10 steps — server/su2_run.py's background thread
greps for these exact prefixes to drive the live progress/ETA the app
shows while a run is going, so this format is load-bearing, not cosmetic.

Must run as its own process (not imported into the FastAPI/uvicorn
process) — pysu2's CDriver and MPI both own real global state that
doesn't belong inside a long-lived web server.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cca_waveform import inlet_velocity_ms, mean_velocity_ms  # noqa: E402
from windkessel import Windkessel, calibrate_two_outlet_rcr  # noqa: E402

COUPLING_RELAXATION = 0.03  # was 0.1 — see the real divergence this under-relaxes further against, below
# Explicit (one-step-behind) Windkessel<->CFD coupling, by construction:
# q_ica/q_eca below are the flow measured at the *end of the previous*
# iteration, driving the pressure applied for *this* one — a real,
# documented source of instability for this pipeline (loop gain
# dominated by Rp), not something under-relaxation alone fully removes
# for every geometry. CONFIRMED on a real case: applied outlet pressure
# drifted monotonically from a correct ~93/63 mmHg start to -106/-40
# mmHg by the end of a 2-cycle run, not a bounded pulsatile oscillation.
# A hard physiological clamp on the *applied* pressure can't fix a
# diverging loop's root cause, but it keeps one from running away to a
# meaningless result silently — if the clamp is doing anything, the
# coupling is unstable for this case and the result should be treated as
# suspect, not just "a bit off".
P_APPLIED_MIN_PA = 20.0 * 133.322   # ~20 mmHg — below normal diastolic, not venous
P_APPLIED_MAX_PA = 220.0 * 133.322  # ~220 mmHg — above any realistic systolic


def _progress(tag, **info):
    print(f"{tag} {json.dumps(info)}", flush=True)


def marker_area_and_flow(driver, marker_id, vel_idx, comm):
    """MPI halo-safe: only sums over nodes this rank actually owns
    (GetNodeDomain), then all-reduces — a marker's nodes are split
    across ranks, and double-counting shared/halo nodes would otherwise
    silently inflate area and flow."""
    from mpi4py import MPI
    area = 0.0
    flow = 0.0
    if marker_id >= 0:
        n_nodes = driver.GetNumberMarkerNodes(marker_id)
        prims = driver.MarkerPrimitives(marker_id)
        for iv in range(n_nodes):
            i_point = driver.GetMarkerNode(marker_id, iv)
            if not driver.GetNodeDomain(i_point):
                continue
            normal = driver.GetMarkerVertexNormals(marker_id, iv, False)
            a = float(np.linalg.norm(normal))
            area += a
            vel = [prims(iv, vel_idx[d]) for d in range(len(vel_idx))]
            flow += sum(v * n for v, n in zip(vel, normal))
    area = comm.allreduce(area, op=MPI.SUM)
    flow = comm.allreduce(flow, op=MPI.SUM)
    return area, flow


def main():
    from mpi4py import MPI
    import pysu2

    cfg = sys.argv[1] if len(sys.argv) > 1 else "run.cfg"
    params_path = sys.argv[2] if len(sys.argv) > 2 else "run_params.json"
    P = json.load(open(params_path))

    psv_cms, edv_cms = P["psv_cms"], P["edv_cms"]
    cycle_s = P["cardiac_cycle_s"]
    ica_fraction = P["ica_fraction"]
    ica_map_pa = P["ica_map_pa"]
    eca_map_pa = P["eca_map_pa"]
    inlet_dir = np.array(P["inlet_direction"], dtype=np.float64)
    inlet_dir /= np.linalg.norm(inlet_dir)

    comm = MPI.COMM_WORLD
    is_root = comm.Get_rank() == 0

    try:
        driver = pysu2.CSinglezoneDriver(cfg, 1, comm)
    except Exception as e:
        if is_root:
            _progress("SU2ERROR", message=f"driver init failed: {e}")
        raise

    marker_ids = driver.GetMarkerIndices()
    id_cca = marker_ids.get("cca_inlet", -1)
    id_ica = marker_ids.get("ica_outlet", -1)
    id_eca = marker_ids.get("eca_outlet", -1)

    prim_idx = driver.GetPrimitiveIndices()
    vel_idx = [prim_idx["VELOCITY_X"], prim_idx["VELOCITY_Y"], prim_idx["VELOCITY_Z"]]

    dt = driver.GetUnsteadyTimeStep()
    total_steps = driver.GetNumberTimeIter()

    inlet_area, _ = marker_area_and_flow(driver, id_cca, vel_idx, comm)
    mean_v = mean_velocity_ms(cycle_s, psv_cms, edv_cms)
    mean_flow = mean_v * inlet_area

    q_ica_mean = mean_flow * ica_fraction
    q_eca_mean = mean_flow * (1.0 - ica_fraction)
    rcr = calibrate_two_outlet_rcr(q_ica_mean, q_eca_mean, ica_map_pa, eca_map_pa)
    wk_ica = Windkessel(*rcr["ica"], P_init=ica_map_pa)
    wk_eca = Windkessel(*rcr["eca"], P_init=eca_map_pa)

    if is_root:
        _progress(
            "SU2PROGRESS", step=0, total_steps=int(total_steps), t_sim=0.0, cycle_s=cycle_s,
            q_ica_mlps=q_ica_mean * 1e6, q_eca_mlps=q_eca_mean * 1e6,
            p_ica_mmhg=ica_map_pa / 133.322, p_eca_mmhg=eca_map_pa / 133.322,
            inlet_area_mm2=inlet_area * 1e6,
        )

    n_cca = driver.GetNumberMarkerNodes(id_cca) if id_cca >= 0 else 0
    p_ica_applied = ica_map_pa
    p_eca_applied = eca_map_pa
    t_start = time.time()

    for time_iter in range(total_steps):
        t = time_iter * dt

        v_mag = inlet_velocity_ms(t, cycle_s, psv_cms, edv_cms)
        for iv in range(n_cca):
            driver.SetMarkerCustomInletFlowVar1(id_cca, iv, v_mag)
            driver.SetMarkerCustomInletFlowDirection(id_cca, iv, list(inlet_dir))

        _, q_ica = marker_area_and_flow(driver, id_ica, vel_idx, comm)
        _, q_eca = marker_area_and_flow(driver, id_eca, vel_idx, comm)
        p_ica_raw = wk_ica.step(dt, q_ica)
        p_eca_raw = wk_eca.step(dt, q_eca)
        p_ica_applied += COUPLING_RELAXATION * (p_ica_raw - p_ica_applied)
        p_eca_applied += COUPLING_RELAXATION * (p_eca_raw - p_eca_applied)
        p_ica_applied = min(max(p_ica_applied, P_APPLIED_MIN_PA), P_APPLIED_MAX_PA)
        p_eca_applied = min(max(p_eca_applied, P_APPLIED_MIN_PA), P_APPLIED_MAX_PA)
        # Clamping only the *applied* pressure would still let the
        # Windkessel's own internal state (wk_ica.P/wk_eca.P) drift
        # unbounded underneath it — reset it back in sync so next step's
        # relaxation starts from the pressure actually imposed on the
        # flow, not a runaway internal value the clamp is silently
        # overriding every single step.
        wk_ica.P = p_ica_applied - wk_ica.Rp * q_ica
        wk_eca.P = p_eca_applied - wk_eca.Rp * q_eca
        if id_ica >= 0:
            driver.SetMarkerCustomOutletPressure(id_ica, p_ica_applied)
        if id_eca >= 0:
            driver.SetMarkerCustomOutletPressure(id_eca, p_eca_applied)

        driver.BoundaryConditionsUpdate()
        driver.Preprocess(time_iter)
        driver.Run()
        driver.Postprocess()
        driver.Update()
        driver.Monitor(time_iter)
        driver.Output(time_iter)

        if is_root:
            _progress(
                "SU2PROGRESS", step=time_iter + 1, total_steps=int(total_steps),
                t_sim=t, cycle_s=cycle_s, elapsed_s=time.time() - t_start,
                v_cca_cms=v_mag * 100, q_ica_mlps=q_ica * 1e6, q_eca_mlps=q_eca * 1e6,
                p_ica_mmhg=p_ica_applied / 133.322, p_eca_mmhg=p_eca_applied / 133.322,
            )

    driver.Finalize()
    if is_root:
        _progress("SU2DONE", total_steps=int(total_steps), elapsed_s=time.time() - t_start)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SU2ERROR {json.dumps({'message': str(e)})}", flush=True)
        raise
