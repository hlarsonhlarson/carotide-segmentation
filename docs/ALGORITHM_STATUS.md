# Algorithm implementation status

This tracks implementation, not clinical validation. Synthetic tests do not
establish accuracy on patient CTA.

## Implemented in the current working tree

- DICOM sorting uses the slice-normal projection of ImagePositionPatient.
- Classic stacks and enhanced multi-frame functional groups are checked for
  duplicate positions, missing/irregular slice steps, mixed orientations,
  inconsistent pixel sizes, non-spatial frame order and sheared grids before
  decoding. Unsupported grids are rejected instead of averaging their spacing.
- Loaded cases retain the DICOM direction matrix. NIfTI input to the model
  includes that direction, spacing and origin.
- Returned canonical labels are resampled onto the original input grid using
  nearest-neighbor interpolation.
- Projects record the XYZ-to-LPS affine and a patient-space copy of centerlines.
  Existing viewer and editing coordinates remain legacy ZYX millimetres.
- Arc-length resampling interpolates at the requested interval instead of
  skipping original vertices.
- New contour series store rotation-minimizing frames. Image sampling and
  both automatic and manual contour conversion use those same frames.
  Old series without frames retain their previous sampling basis.
- The CTA radial half-height method measures lumen contours and rejects missing
  edges, unenhanced centers and bright outer peaks. It is available in the
  editor and used for automatic CTA drafts.
- Failed automatic 2D segmentation no longer silently inserts a circle.
- The editor displays lumen area, equivalent diameter and minimum/maximum
  Feret diameters. A diameter-ratio NASCET estimate requires explicit operator
  selection of a normal distal ICA reference and an ICA lesion. Invalid rings
  and estimated circle contours cannot be used. This is not a validated
  clinical NASCET measurement method or an automatic near-occlusion detector.
- A longitudinal profile plots area at each measured section, with explicit
  gaps and navigation to the minimum area or minimum Feret diameter in a
  selected range. Minima are not automatically classified as ICA stenoses.
- Trapezoidal section integrals estimate lumen, annular and high-density volumes
  over covered intervals only. Outputs include covered/selected lengths and
  excluded intervals. Missing contours, estimated circles, excessive sample
  gaps and tangent changes above 30 degrees are excluded. High-density totals
  require identical HU/component-size parameters throughout the selected range.
  These are sampled estimates, not voxel-mask volumes or plaque composition.
- Batch outer-wall detection processes unfilled sections, retains existing
  boundaries, reports failed sections and can stop after the current request.
  Outer-wall annotations can be removed independently of the lumen.
- Worker inference runs outside the event loop, rejects concurrent jobs, and
  configures PyTorch threading once per process.
- HTTP errors from the worker preserve their actual details.
- CFD averaging interpolates the exact start of the final recorded cardiac
  period instead of dropping its boundary sample. Partial runs are explicitly
  identified. Two full cycles are compared using phase-aligned global RMS of
  WSS vectors and pressure, with absolute differences and a stated 5% tolerance.
  Source intervals larger than one tenth of a cycle prevent a positive
  periodicity assessment. Agreement is not a claim of solver/mesh validation.
- CFD extraction verifies wall node IDs, coordinates/units and field dimensions
  on every snapshot. Invalid/missing/nonfinite fields and mismatched node
  ordering are rejected. OSI at zero WSS has an explicit validity mask.
- The SU2 boundary-normal sign is corrected: the wrapper returns inward
  normals, which must be negated for outward flux. This was verified against
  the installed SU2 source and a runtime duct-normal probe. The correction
  applies both to RCR input flows and diagnostics.
- The runner records same-state inlet/outlet flux estimates after each solve,
  reverse-flow indicators and pressure-clamp events in `boundary_flows.csv`.
  Postprocessing reports last-cycle integrated volumes and absolute imbalance;
  the estimate uses boundary nodal velocities, not conservative numerical
  fluxes from the solver, so it is not a residual convergence certificate.

## Remaining implementation and validation

0. Complete migration of legacy annotations/surfaces to LPS; resampling or
   explicit contiguous-stack selection for rejected irregular/sheared DICOM
   series; optional isotropic processing grids restricted to ROI.
1. Validate Fast seeds and component filtering on representative CTA.
2. Validate ICA/ECA labeling and endpoint selection; implement explicit
   bifurcation uncertainty instead of assuming every tree is anatomically valid.
3. Expose MPR sampling parameters and verify all real-case reformats.
4. Validate the radial method on stenoses; longitudinal intensity reference and
   confidence-aware smoothing must preserve genuine diameter minima.
5–7. Experimental sectional outer-wall proposals use cyclic radial optimization,
   edge-support and search-limit rejection. The editor overlays the proposed
   boundary and high-density components with an explicit CTA HU threshold;
   results/parameters persist and are invalidated by lumen edits.
   Manual outer-wall tracing is supported when detection fails, with checks
   for nested nonintersecting rings. Manual results carry no detector confidence.
   Sectional areas remain 2D; longitudinal integrals are coverage-limited
   estimates, not validated calcium volumes. Plaque-component classification,
   longitudinal wall regularization and expert validation remain pending.
   Contrast CTA must not use 130 HU as an unconditional calcium threshold.
   Estimated wall thickness must not be represented as a measured boundary.
   The implemented NASCET estimate needs validation against expert measurements;
   it requires an explicitly selected normal distal ICA reference.
8. Ophthalmic artery tracing and optional outlet support are NOT implemented.
9. Surface/volume meshing exists; prism boundary layers and three-level mesh
   convergence are not established by this change.
10. SU2 export/run/postprocessing and sampled cycle comparison exist. Outlet
    calibration, mass balance, mesh/time-step convergence and real-run periodic
    convergence remain unverified. The existing 0.5 skin-friction-to-WSS scale
    remains a project convention pending independent dimensional verification.
    The web interface can now proxy a native macOS API with verified
    SU2/pysu2/MPI imports and custom-boundary methods. A four-step synthetic
    duct run completed through the normal solver and postprocessing chain.
    Subsequent 600-step duct runs with corrected flux signs showed no reverse
    flow in the last cycle and about 2% cycle-to-cycle WSS RMS differences,
    but boundary nodal-flux imbalance remained about 23% (20 inner iterations)
    and 20% (80 inner iterations). These runs do not establish mass conservation
    or physiological validity; conservative boundary-flux extraction and mesh/
    boundary-condition verification remain open.
    This establishes execution/output compatibility, not patient CFD accuracy
    or periodic convergence. See `docs/NATIVE_CFD.md` for startup and rollback.
11. Expert-reference Dice/HD95, observer repeatability and CFD sensitivity runs
    require reference segmentations and completed simulation experiments.

## Checks

`python -m pytest -q tests server/tests` and `npm run build` in `frontend`.
Geometry tests include asymmetric labels through NIfTI canonicalization,
rotated/reflected LPS transforms and pixel-center MPR consistency.

Header-only checks of the local studies found regular grids in the 379-slice
and 238-slice series. The 269-file series has 267 intervals of 1 mm and one
interval of 59 mm; it is rejected until a contiguous stack is explicitly
selected or the source series is corrected. This check does not establish
segmentation accuracy.
