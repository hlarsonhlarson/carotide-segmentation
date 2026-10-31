"""Builds a mesh-ready, un-capped surface from accepted per-point Series
segmentations (SimVascular's lofting step, reimplemented) — rasterizes
each branch's sequence of real segmented cross-sections (point, tangent,
contour) into a shared volume mask by sweeping each contour's actual
shape along the path (not an idealized circle at some fitted radius —
stamping circles was tried first and produced a visible ring-like
corrugation on the exported surface that the contour-ribbon 3D preview,
built from this same data, never showed), then reuses
carotid_seg/meshing.py's
already-proven resample->marching-cubes->smooth pipeline (proven on 20+
real patients earlier this session) to extract a smooth, watertight
surface — a voxel-level union handles the CCA/ICA/ECA bifurcation
automatically, the same way it did for the earlier TotalSegmentator-driven
pipeline, without needing an explicit bifurcation-surface stitch.

Deliberately left uncapped (inlet/outlet faces stay open, no end caps) —
unlike the earlier TotalSegmentator-driven pipeline, which built on a mask
that ran further than wanted and had to find and cut where each branch
should end. CONFIRMED (not assumed) that marching cubes never leaves this
opening on its own, even when the mask itself is only ever built exactly
as far as segmented: vtkDiscreteFlyingEdges3D always finds the full
boundary of a binary mask's "inside" region, so every stamped end,
however short the mask runs, comes out as a fully closed cap — an
explicit cut is the only way to open one. See _local_open_end for how
that cut is done (and why not with an infinite plane).
"""
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carotid_seg.meshing import (
    _resample_isotropic, _numpy_to_vtk_image, _largest_component,
    count_boundary_loops, write_stl,
)

# carotid_seg.meshing's own _smooth (150 iterations, passband 0.01) is
# tuned for that pipeline's coarser, non-isotropic-resampled
# TotalSegmentator-driven mask (proven on 20+ real patients there) — reused
# here at first since it's already proven elsewhere, but CONFIRMED (on a
# real patient's ECA, tapering to ~1.3mm radius — only about 5-6 voxels
# across at this module's isotropic resample resolution) that smoothing
# this aggressively collapses a branch that thin into a flat fin rather
# than rounding it: verified by comparing the raw (unsmoothed) marching-
# cubes mesh, which still has real cross-sectional volume there, against
# the smoothed one, which doesn't, at the exact same point. A real vessel
# this thin needs much gentler smoothing to survive; this module's own
# contour-based mask is already far cleaner going in (real segmented
# shapes swept along the path, not a blocky organ-segmentation mask), so
# it doesn't need nearly as much passes to begin with.
SMOOTHING_ITERATIONS = 60
SMOOTHING_PASSBAND = 0.05


def _smooth(poly):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(SMOOTHING_ITERATIONS)
    smoother.SetPassBand(SMOOTHING_PASSBAND)
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(smoother.GetOutput())
    return out

DISK_HALF_THICKNESS_MM = 1.2
# A stamped disk only covers +/- DISK_HALF_THICKNESS_MM along its own
# tangent, so two consecutive disks only touch if that reaches at least
# half the true center-to-center spacing between them. CONFIRMED (on a
# real patient's ICA series) that real path spacing varies point to point
# and reaches ~3mm even though it averages ~2.5mm — comfortably more than
# 2*DISK_HALF_THICKNESS_MM=2.4mm, which left a genuine gap in the
# rasterized mask at exactly that point, and that gap survives isotropic
# SDF resampling and marching cubes regardless of the SDF zoom step's
# interpolation order (confirmed by testing both cubic and linear) — it's
# a real topological hole in the mask, not a resampling artifact, and
# vtkPolyDataConnectivityFilter's first largest-component pass (right
# after marching cubes, before any smoothing) then discards the shorter
# side of the break entirely, reading as the branch being cut short
# partway along, or (for a short branch like the ECA) as a thin, nearly
# degenerate sliver.
#
# A first fix thickened every disk off the series' own max spacing
# instead of this fixed value — closed that gap, but CONFIRMED (on a
# real patient's ECA, tapering to ~1.3mm radius right where the path also
# curves sharply approaching the bifurcation) it introduces a *worse*
# failure on a thin, sharply curving branch: once a disk's own thickness
# along tangent exceeds the vessel's own radius there, consecutive disks
# stamped in neighboring-but-differently-rotated cross-section planes no
# longer union into anything tube-shaped — their union reads as a
# twisted flat fin, not a cylinder, and strong smoothing afterward
# collapses it further rather than rounding it out. Bridging the same gap
# with *more, thin* interpolated disks (_bridge_gap) instead of *fewer,
# thick* ones closes it via sampling density along the path instead of
# via each disk's own thickness, so no single stamped disk is ever
# thicker than a real vessel is wide.
BRIDGE_STEP_MM = DISK_HALF_THICKNESS_MM * 1.5
# _bridge_gap's step count was originally driven only by raw center-to-
# center distance (above) — enough to avoid a topological gap in the
# mask, but CONFIRMED (on a real patient's ICA at its first sharp bend,
# where the tangent rotates 50+ degrees between consecutive path points
# only ~2.2mm apart — well under BRIDGE_STEP_MM already, so distance-
# based bridging added at most one sub-disk there) that even a single
# well-placed intermediate disk isn't enough when the *orientation*
# between consecutive cross-sections swings that far: each disk is a
# flat slab perpendicular to its own tangent, so a handful of disks
# whose planes fan out sharply from each other union into a faceted,
# lobed knot at the bend rather than a smooth swept tube — more, thinner
# disks only fixes this if they're also closely spaced in *rotation*,
# not just in position. Capping how much the tangent is allowed to turn
# between consecutive stamped disks (same bridging mechanism, just also
# triggered by angle) makes the swept union converge to a smooth tube
# through a sharp bend the same way a polygon converges to a circle as
# its segment count grows.
BRIDGE_MAX_ANGLE_DEG_PER_STEP = 10.0
BBOX_MARGIN_MM = 10.0
CUT_INSET_MM = 4.0
# Comfortably larger than any realistic carotid/ICA/ECA cross-section
# (~2-4mm radius) so the local open-end cut always fully opens the face,
# but far smaller than how close together two genuinely distinct loops
# of even a tightly tortuous vessel actually get (12mm+, confirmed via a
# synthetic kinked-path test) — so it can't reach one.
LOCAL_CUT_RADIUS_MM = 9.0
# LOCAL_CUT_RADIUS_MM alone is sized for a CCA/ICA-scale trunk (~3-4mm
# radius); CONFIRMED (on a real patient's ECA, tapering to ~1.3-1.9mm
# radius at its own distal cut point) that using that same fixed 9mm for
# a noticeably thinner branch's own outlet cut removes most of the
# branch, not just its end cap — the bounding sphere is 5-7x the vessel's
# own radius there, large enough to eat back along a short, already-thin
# tube almost to the bifurcation, leaving a collapsed sliver instead of a
# tube. Scaling the cut radius off that point's own fitted radius_mm
# keeps it proportional to whatever vessel it's actually cutting; the
# LOCAL_CUT_RADIUS_MM ceiling keeps a larger-caliber cut (the shared
# inlet, or any branch with a wide outlet) exactly at today's already-
# proven behavior, and the floor keeps a very thin tip's cut from
# shrinking to the point it fails to fully open the face.
CUT_RADIUS_FACTOR = 2.5
MIN_CUT_RADIUS_MM = 2.0


def _inset_series(series, inset_mm, at_start: bool):
    """Drops the points within inset_mm of arc length of the given end,
    before rasterizing — so the mask's own extent already stops at a
    clean, full-diameter cross-section instead of right at the last
    stamped disk, which is thinner and more tapered (every disk is a
    half_thickness_mm-thick slab; the very last one only has material on
    one side of it). CONFIRMED this alone does *not* leave an opening —
    marching cubes caps every end of a binary mask's boundary regardless
    of how short the mask runs — so _local_open_end still needs to
    explicitly cut here afterward; this just gives that cut a clean,
    full-diameter place to do it."""
    ordered = series if at_start else list(reversed(series))
    acc = 0.0
    cut_idx = len(ordered)
    for i in range(1, len(ordered)):
        acc += float(np.linalg.norm(np.asarray(ordered[i]["point_mm"]) - np.asarray(ordered[i - 1]["point_mm"])))
        if acc >= inset_mm:
            cut_idx = i
            break
    trimmed = ordered[cut_idx:]
    return trimmed if at_start else list(reversed(trimmed))


def _xyz(point_mm):
    """This module's output surface's vertex coordinates are in VTK
    (x,y,z) order, the reverse of this app's own [z,y,x] point_mm
    convention (_numpy_to_vtk_image builds the marching-cubes input image
    with spacing/origin reversed) — any world point has to go through
    this same flip to land in the surface's own coordinate space (see
    server/su2_export.py's identical helper, for the identical reason)."""
    return [point_mm[2], point_mm[1], point_mm[0]]


def _local_open_end(poly, cut_point_mm, cut_tangent_mm, bound_radius_mm, at_start: bool):
    """Opens a face in `poly` by deleting triangles that are both past
    the cut plane (through cut_point_mm, normal cut_tangent_mm) *and*
    within bound_radius_mm of cut_point_mm — bounding the cut keeps it
    from reaching a distant, unrelated loop of the same curving vessel,
    which an *infinite*-plane clip (carotid_seg.meshing._clip_away, used
    by the earlier version of this cut) could also slice through; its
    "keep the larger piece" rule to recover from that can end up keeping
    the wrong side of that unrelated cut instead of the correct side of
    the intended one, discarding a real connected stretch of the vessel
    elsewhere — read as the model being cut off partway along its length,
    not cleanly opened at its tip. Bounding the cut to a small sphere
    around the actual cut point means there is nothing *to* reach
    outside it: a distant self-proximate loop is left completely
    untouched rather than needing a size-based rule to recover from."""
    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    origin = np.asarray(_xyz(cut_point_mm), dtype=np.float64)
    normal = np.asarray(_xyz(cut_tangent_mm), dtype=np.float64)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    # The tangent points *forward* along the path (toward the far/distal
    # end). At the trimmed series' own first point (the inlet), the only
    # remaining material beyond it is *behind* (the backward, -tangent
    # side) — the thin cap from that first stamped disk's own half-
    # thickness. At its own last point (a distal outlet), the cap is
    # ahead instead (+tangent). Flip which side counts as "past the cut"
    # accordingly.
    if at_start:
        normal = -normal
    rel = pts - origin
    along = rel @ normal
    dist = np.linalg.norm(rel, axis=1)
    remove_point = (along > 0) & (dist < bound_radius_mm)
    if not remove_point.any():
        return poly

    polys = poly.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    new_polys = vtk.vtkCellArray()
    while polys.GetNextCell(id_list):
        ids = [id_list.GetId(k) for k in range(id_list.GetNumberOfIds())]
        if any(remove_point[i] for i in ids):
            continue
        new_polys.InsertNextCell(len(ids))
        for i in ids:
            new_polys.InsertCellPoint(i)

    result = vtk.vtkPolyData()
    result.SetPoints(poly.GetPoints())
    result.SetPolys(new_polys)
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(result)
    cleaner.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def _close_spurious_holes(poly, protect_points_xyz):
    """Closes every boundary loop *except* the ones nearest each known
    real cut point (the inlet and each branch's own outlet, already
    opened by _local_open_end) by fan-triangulating it from its own
    centroid. Replaces an earlier approach (vtkFillHolesFilter, tried at
    increasing HoleSize thresholds until the loop count dropped to
    n_expected) that CONFIRMED (via a real patient's ICA, by locating
    each post-fill boundary loop's centroid) could close the wrong hole —
    it isn't aware of which loops are real outlets, only of their size,
    and VTK's own "hole size" metric (a circumsphere radius) doesn't
    reliably track which hole is actually smaller once holes are
    irregular rather than circular: it closed a genuine ~5mm-wide outlet
    while leaving a true sub-millimeter stray pinhole elsewhere still
    open. A simple proximity *radius* around each cut point isn't enough
    either — CONFIRMED a real spurious pinhole can sit nearer to a real
    outlet than the outlet's own (irregular, off-axis) boundary loop
    centroid does, so both fell inside the same radius and both stayed
    open. Matching each cut point to its single *nearest* loop (not
    everything within some radius) is what correctly tells the two
    apart."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    edges = fe.GetOutput()
    if edges.GetNumberOfPoints() == 0:
        return poly

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(edges)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_regions = conn.GetNumberOfExtractedRegions()

    region_polys = []
    centroids = []
    for r in range(n_regions):
        region_filter = vtk.vtkPolyDataConnectivityFilter()
        region_filter.SetInputData(edges)
        region_filter.SetExtractionModeToSpecifiedRegions()
        region_filter.AddSpecifiedRegion(r)
        region_filter.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(region_filter.GetOutput())
        cleaner.Update()
        region_poly = cleaner.GetOutput()
        region_polys.append(region_poly)
        if region_poly.GetNumberOfPoints() == 0:
            centroids.append(None)
            continue
        rpts = numpy_support.vtk_to_numpy(region_poly.GetPoints().GetData())
        centroids.append(rpts.mean(axis=0))

    protected_regions = set()
    for pp in protect_points_xyz:
        candidates = [(i, c) for i, c in enumerate(centroids) if c is not None]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda ic: np.linalg.norm(ic[1] - pp))
        protected_regions.add(nearest[0])

    append = vtk.vtkAppendPolyData()
    append.AddInputData(poly)

    for r in range(n_regions):
        if r in protected_regions:
            continue
        region_poly = region_polys[r]
        if region_poly.GetNumberOfPoints() == 0:
            continue
        centroid = centroids[r]

        stripper = vtk.vtkStripper()
        stripper.SetInputData(region_poly)
        stripper.Update()
        strip = stripper.GetOutput()
        lines = strip.GetLines()
        if lines.GetNumberOfCells() == 0:
            continue
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        lines.GetNextCell(id_list)
        ordered_ids = [id_list.GetId(k) for k in range(id_list.GetNumberOfIds())]
        if len(ordered_ids) < 3:
            continue
        strip_pts = numpy_support.vtk_to_numpy(strip.GetPoints().GetData())
        loop_coords = strip_pts[ordered_ids]
        if np.allclose(loop_coords[0], loop_coords[-1]):
            loop_coords = loop_coords[:-1]
        n = len(loop_coords)
        if n < 3:
            continue

        patch_points = vtk.vtkPoints()
        for c in loop_coords:
            patch_points.InsertNextPoint(*c)
        centroid_id = patch_points.InsertNextPoint(*centroid)
        patch_polys = vtk.vtkCellArray()
        for i in range(n):
            j = (i + 1) % n
            patch_polys.InsertNextCell(3)
            patch_polys.InsertCellPoint(i)
            patch_polys.InsertCellPoint(j)
            patch_polys.InsertCellPoint(centroid_id)
        patch = vtk.vtkPolyData()
        patch.SetPoints(patch_points)
        patch.SetPolys(patch_polys)
        append.AddInputData(patch)

    append.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(append.GetOutput())
    cleaner.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def _cut_radius_mm(point) -> float:
    r = point.get("radius_mm") or 0.0
    return float(np.clip(r * CUT_RADIUS_FACTOR, MIN_CUT_RADIUS_MM, LOCAL_CUT_RADIUS_MM))


def _series_bbox_mm(series_by_branch: dict):
    pts = [p["point_mm"] for series in series_by_branch.values() for p in series]
    arr = np.asarray(pts)
    return arr.min(axis=0), arr.max(axis=0)


def _max_radius_mm(series_by_branch: dict) -> float:
    radii = [p.get("radius_mm") or 0.0 for series in series_by_branch.values() for p in series]
    return max(radii) if radii else 0.0


def _plane_basis(tangent_mm):
    """Orthonormal (t, u, v) for the oblique cross-section plane at a path
    point — t is the (normalized) tangent, u/v span the plane itself. Not
    required to be the exact basis reslice.py/segment2d.py used to extract
    the contour in the first place: projecting the same contour and the
    same query voxels into any shared orthonormal basis for that plane
    gives an equivalent point-in-polygon test."""
    t = np.asarray(tangent_mm, dtype=np.float64)
    t = t / max(np.linalg.norm(t), 1e-9)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, arbitrary)
    u = u / max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u)
    return t, u, v


def _points_in_polygon_2d(px, py, poly_x, poly_y):
    """Vectorized even-odd ray-casting point-in-polygon test. px/py: query
    point coordinates (any matching shape, e.g. a voxel grid); poly_x/
    poly_y: 1D arrays of the polygon's own vertices, in order."""
    n = len(poly_x)
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        dy = yj - yi
        dy = dy if abs(dy) > 1e-12 else 1e-12
        crosses = (yi > py) != (yj > py)
        x_intersect = (xj - xi) * (py - yi) / dy + xi
        inside ^= crosses & (px < x_intersect)
        j = i
    return inside


def _stamp_contour(mask, crop_origin_mm, spacing_mm, center_mm, tangent_mm, contour_mm,
                    half_thickness_mm=DISK_HALF_THICKNESS_MM):
    """Rasterizes the *actual* segmented cross-section — not an idealized
    circle at some fitted radius — into `mask`, as a short slab of
    thickness half_thickness_mm either side of the point. A real vessel
    wall's shape already varies smoothly point to point (confirmed by the
    ribbon-lofted 3D preview built from this exact same contour data,
    which looks right): collapsing every point to one scalar radius
    before stamping a circle was a lossy step that threw that shape away,
    and was the real source of a ring-like corrugation seen on exported
    STLs even though the preview looked fine — same data, a stamped
    circle per point just isn't the same surface as the real contours."""
    center = np.asarray(center_mm, dtype=np.float64)
    contour = np.asarray(contour_mm, dtype=np.float64)
    if len(contour) < 3:
        return False
    t, u, v = _plane_basis(tangent_mm)

    rel = contour - center
    poly_u = rel @ u
    poly_v = rel @ v
    max_r = float(np.max(np.hypot(poly_u, poly_v)))
    if max_r <= 0:
        return False

    margin = max_r + half_thickness_mm + float(np.max(spacing_mm))
    lo_vox = np.floor((center - margin - crop_origin_mm) / spacing_mm).astype(int)
    hi_vox = np.ceil((center + margin - crop_origin_mm) / spacing_mm).astype(int)
    lo_vox = np.maximum(lo_vox, 0)
    hi_vox = np.minimum(hi_vox, np.asarray(mask.shape))
    if np.any(hi_vox <= lo_vox):
        return False

    zz, yy, xx = np.mgrid[lo_vox[0]:hi_vox[0], lo_vox[1]:hi_vox[1], lo_vox[2]:hi_vox[2]]
    world = crop_origin_mm + np.stack([zz, yy, xx], axis=-1) * np.asarray(spacing_mm)
    delta = world - center
    along = delta @ t
    within_thickness = np.abs(along) <= half_thickness_mm
    pu = delta @ u
    pv = delta @ v
    inside_poly = _points_in_polygon_2d(pu, pv, poly_u, poly_v)
    mask[lo_vox[0]:hi_vox[0], lo_vox[1]:hi_vox[1], lo_vox[2]:hi_vox[2]] |= (within_thickness & inside_poly)
    return True


def _contour_radius_and_offset(contour_mm, center_mm, u, v):
    """Area-equivalent radius (shoelace, in the point's own cross-section
    plane) and how far the contour's own centroid sits from the path
    point — both computed directly from the real contour, used only to
    *flag* a point as untrustworthy, never to replace its shape."""
    contour = np.asarray(contour_mm, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64)
    rel = contour - center
    pu = rel @ u
    pv = rel @ v
    area = 0.5 * abs(np.dot(pu, np.roll(pv, 1)) - np.dot(pv, np.roll(pu, 1)))
    radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    centroid_offset = float(np.hypot(pu.mean(), pv.mean()))
    return radius, centroid_offset


def _reoriented_contour(neighbor_point_mm, neighbor_contour_mm, neighbor_tangent_mm,
                         target_point_mm, target_tangent_mm):
    """Re-expresses a neighbor's contour shape in the *target* point's own
    cross-section plane, not just translated to the target's location. A
    plain translation leaves the contour planar in the *neighbor's*
    plane — fine only if the two tangents happen to point the same way,
    which is exactly not the case where substitution is needed most (a
    sharp turn, e.g. right at the bifurcation): stamping a neighbor-plane
    contour against the target's own (different) tangent skews it,
    producing a sharp spike or a hole instead of a smooth substitute
    cross-section. Preserves the contour's own 2D shape (its coordinates
    in the neighbor's own u/v) and just re-anchors that shape to the
    target's point and plane."""
    _, nu, nv = _plane_basis(neighbor_tangent_mm)
    _, tu, tv = _plane_basis(target_tangent_mm)
    rel = np.asarray(neighbor_contour_mm, dtype=np.float64) - np.asarray(neighbor_point_mm, dtype=np.float64)
    local_u = rel @ nu
    local_v = rel @ nv
    target_center = np.asarray(target_point_mm, dtype=np.float64)
    return (target_center + local_u[:, None] * tu + local_v[:, None] * tv).tolist()


def _radius_by_angle(local_u, local_v, n_angles):
    """Resamples a ring (already expressed as local_u/local_v around its
    own center) to n_angles evenly-spaced angle samples, each holding the
    ring's own radius at that angle — via linear interpolation over angle
    (periodic wraparound), so two rings with different point counts or
    starting phase become directly comparable sample-for-sample. Assumes
    a star-shaped contour around its center (true for a real vessel
    cross-section; the shapes this module stamps already are)."""
    angle = np.arctan2(local_v, local_u)
    radius = np.hypot(local_u, local_v)
    order = np.argsort(angle)
    angle, radius = angle[order], radius[order]
    # extend periodically so np.interp wraps correctly at the -pi/pi seam
    angle_ext = np.concatenate([angle - 2 * np.pi, angle, angle + 2 * np.pi])
    radius_ext = np.concatenate([radius, radius, radius])
    query = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return np.interp(query, angle_ext, radius_ext), query


def _blend_contours_by_angle(point_a_mm, contour_a_mm, tangent_a_mm,
                              point_b_mm, contour_b_mm, tangent_b_mm,
                              target_point_mm, target_tangent_mm, t, n_angles=48):
    """Builds the sub-point's own contour as a genuine angle-by-angle
    interpolation between the two real endpoints' shapes, both first
    re-expressed in the *target*'s own cross-section plane (via the same
    re-anchoring _reoriented_contour uses) so they're directly
    comparable. Replaces an earlier version that hard-switched between
    a straight copy of contour_a (t<0.5) or contour_b (t>=0.5) — CONFIRMED
    (on a real patient's ICA, where consecutive points' fitted radius
    swings 3.35 -> 2.50 -> 3.25 -> 2.59mm within a handful of points at a
    sharp bend, from locally noisy segmentation rather than real
    anatomy) that a hard switch stamps a sudden full-size jump at the
    bridge's own midpoint instead of tapering through it, which reads as
    exactly the same kind of lobed knot bridging was meant to smooth
    away. A smooth per-angle taper spreads any one endpoint's own noise
    across the whole bridge instead of concentrating it at one sharp
    edge."""
    _, nu_a, nv_a = _plane_basis(tangent_a_mm)
    _, nu_b, nv_b = _plane_basis(tangent_b_mm)
    _, tu, tv = _plane_basis(target_tangent_mm)
    target_center = np.asarray(target_point_mm, dtype=np.float64)

    rel_a = np.asarray(contour_a_mm, dtype=np.float64) - np.asarray(point_a_mm, dtype=np.float64)
    radius_a, angles = _radius_by_angle(rel_a @ nu_a, rel_a @ nv_a, n_angles)

    rel_b = np.asarray(contour_b_mm, dtype=np.float64) - np.asarray(point_b_mm, dtype=np.float64)
    radius_b, _ = _radius_by_angle(rel_b @ nu_b, rel_b @ nv_b, n_angles)

    blended_radius = (1 - t) * radius_a + t * radius_b
    local_u = blended_radius * np.cos(angles)
    local_v = blended_radius * np.sin(angles)
    return (target_center + local_u[:, None] * tu + local_v[:, None] * tv).tolist()


def _smooth_contour_shape(contour_mm, window=1):
    """Light moving-average smoothing *around the contour's own closed
    perimeter* (not across path points — see _radius_outliers for that) —
    a real per-point Level Set/Threshold detection on real CT data comes
    back with small-scale jaggedness along its boundary (partial-volume
    effects, a nearby calcification, noise) that isn't wrong enough to
    flag as an outlier point (its overall size and position are fine) but
    still stamps as a visible small dimple or nick once rasterized: the
    ribbon-lofted 3D preview smooths right over single-vertex jitter like
    this just by how ring-to-ring interpolation works, so it never showed
    there, but a voxel stamp of the raw, jagged boundary does. A window=1
    (3-point) moving average removes single-vertex noise without
    flattening the contour toward a circle the way reducing it to a
    scalar radius did."""
    pts = np.asarray(contour_mm, dtype=np.float64)
    n = len(pts)
    if n < 5:
        return contour_mm
    smoothed = np.zeros_like(pts)
    for k in range(-window, window + 1):
        smoothed += np.roll(pts, k, axis=0)
    smoothed /= (2 * window + 1)
    return smoothed.tolist()


def _radius_outliers(radii, window=2, ratio=1.8):
    """Flags points whose area-equivalent radius deviates from the local
    median (over nearby *present* points) by more than `ratio`x either
    way. A real vessel's caliber doesn't jump point-to-point — consecutive
    points are only CONTROL_POINT_SPACING_MM=2mm apart — so an abrupt jump
    is itself the strongest available signal that a point's segmentation
    locked onto the wrong structure (a leak into adjacent tissue) rather
    than real anatomy."""
    n = len(radii)
    flagged = [False] * n
    present = [i for i, r in enumerate(radii) if r > 0]
    for idx, i in enumerate(present):
        lo = max(0, idx - window)
        hi = min(len(present), idx + window + 1)
        neighborhood = [radii[present[k]] for k in range(lo, hi) if present[k] != i]
        if not neighborhood:
            continue
        med = float(np.median(neighborhood))
        if med <= 0:
            continue
        if radii[i] > med * ratio or radii[i] < med / ratio:
            flagged[i] = True
    return flagged


def _bridge_gap(point_a, contour_a, point_b, contour_b, step_mm=BRIDGE_STEP_MM,
                 max_angle_deg_per_step=BRIDGE_MAX_ANGLE_DEG_PER_STEP):
    """Interpolated sub-points to stamp strictly between two consecutive
    path points whose spacing exceeds step_mm, or whose tangent rotates
    by more than max_angle_deg_per_step, so the gap closes through
    stamping *density* rather than through a single disk thicker than the
    vessel itself (see BRIDGE_STEP_MM), and a sharp bend closes through
    stamping disks closely spaced *in orientation* too (see
    BRIDGE_MAX_ANGLE_DEG_PER_STEP) rather than one intermediate disk
    whose plane still fans out sharply from its neighbors. Each
    sub-point's shape borrows whichever real endpoint it's nearer to,
    re-oriented to its own interpolated point/tangent via
    _reoriented_contour — the same plane-aware re-anchoring already used
    for outlier substitution — so a sub-point's disk stays consistent
    with the real cross-section it sits closest to instead of blending
    two independently-segmented shapes."""
    a = np.asarray(point_a["point_mm"], dtype=np.float64)
    b = np.asarray(point_b["point_mm"], dtype=np.float64)
    gap = float(np.linalg.norm(b - a))
    ta = np.asarray(point_a["tangent_mm"], dtype=np.float64)
    ta = ta / max(np.linalg.norm(ta), 1e-9)
    tb = np.asarray(point_b["tangent_mm"], dtype=np.float64)
    tb = tb / max(np.linalg.norm(tb), 1e-9)
    turn_deg = float(np.degrees(np.arccos(np.clip(np.dot(ta, tb), -1.0, 1.0))))

    n_steps = max(
        int(np.ceil(gap / step_mm)),
        int(np.ceil(turn_deg / max_angle_deg_per_step)) if max_angle_deg_per_step > 0 else 1,
    )
    n_steps = min(n_steps, 30)  # safety cap against pathological input, not expected in practice
    if n_steps < 2:
        return []

    sub_points = []
    for k in range(1, n_steps):
        t = k / n_steps
        point_mm = (a + t * (b - a)).tolist()
        tangent = ta + t * (tb - ta)
        norm = np.linalg.norm(tangent)
        tangent_mm = (tangent / norm if norm > 1e-9 else ta).tolist()
        contour_mm = _blend_contours_by_angle(
            point_a["point_mm"], contour_a, point_a["tangent_mm"],
            point_b["point_mm"], contour_b, point_b["tangent_mm"],
            point_mm, tangent_mm, t,
        )
        sub_points.append({"point_mm": point_mm, "tangent_mm": tangent_mm, "contour_mm": contour_mm})
    return sub_points


def _rasterize_branch(mask, crop_origin_mm, spacing_mm, series):
    """For each point with a contour, rasterizes that *real* shape — not
    an idealized circle — except for points flagged as an outlier (wildly
    different caliber than their neighbors, or centered far from the path
    itself: a leak onto the wrong structure, not real anatomy), which
    borrow the *nearest trustworthy neighbor's* real contour shape
    instead, re-anchored to this point's own location *and* cross-section
    plane (see _reoriented_contour — a plain translation isn't enough
    when the two tangents point noticeably different ways, which is
    exactly when substitution is most needed). A point
    that was simply never segmented at all (no contour present) still
    leaves a true gap, same as before — only a point with an actively bad
    detection gets bridged, since leaving consecutive bad points as gaps
    was found (via a synthetic test with injected leaks) to sever the
    mask into disconnected pieces there, losing an entire branch to
    _largest_component."""
    radii = [0.0] * len(series)
    offsets = [0.0] * len(series)
    has_contour = [False] * len(series)
    for i, point in enumerate(series):
        contour = point.get("contour_mm")
        if not contour or len(contour) < 3:
            continue
        has_contour[i] = True
        _, u, v = _plane_basis(point["tangent_mm"])
        radii[i], offsets[i] = _contour_radius_and_offset(contour, point["point_mm"], u, v)

    valid_radii = [r for r in radii if r > 0]
    typical_radius = float(np.median(valid_radii)) if valid_radii else 1.0
    radius_outlier = _radius_outliers(radii)
    is_outlier = [
        has_contour[i] and (radii[i] <= 0 or radius_outlier[i] or offsets[i] > typical_radius * 1.5)
        for i in range(len(series))
    ]
    valid_indices = [i for i in range(len(series)) if has_contour[i] and not is_outlier[i]]

    resolved_contours = [None] * len(series)
    substituted = 0
    for i, point in enumerate(series):
        if is_outlier[i]:
            if not valid_indices:
                continue
            nearest = min(valid_indices, key=lambda j: abs(j - i))
            neighbor = series[nearest]
            resolved_contours[i] = _reoriented_contour(
                neighbor["point_mm"], neighbor["contour_mm"], neighbor["tangent_mm"],
                point["point_mm"], point["tangent_mm"],
            )
            substituted += 1
        elif has_contour[i]:
            resolved_contours[i] = _smooth_contour_shape(point["contour_mm"])

    stamped = 0
    for i, point in enumerate(series):
        contour = resolved_contours[i]
        if contour is None:
            continue
        if _stamp_contour(mask, crop_origin_mm, spacing_mm, point["point_mm"], point["tangent_mm"], contour):
            stamped += 1

    # Bridge consecutive *resolved* points whose spacing is wide enough
    # that their own disks wouldn't touch — see _bridge_gap/BRIDGE_STEP_MM.
    # A point that was never segmented at all (resolved_contours[i] is
    # None, not substituted) still leaves a true gap on each side of it,
    # same as before: only closing the space *between two points that do
    # have a resolved shape* is safe, since a genuinely missing point has
    # no real cross-section to interpolate from.
    resolved_indices = [i for i in range(len(series)) if resolved_contours[i] is not None]
    for idx in range(len(resolved_indices) - 1):
        i, j = resolved_indices[idx], resolved_indices[idx + 1]
        if j != i + 1:
            continue
        for sub in _bridge_gap(series[i], resolved_contours[i], series[j], resolved_contours[j]):
            _stamp_contour(mask, crop_origin_mm, spacing_mm, sub["point_mm"], sub["tangent_mm"], sub["contour_mm"])

    return stamped, substituted


def build_surface(array_shape, spacing_mm, origin_mm, series_by_branch: dict):
    """series_by_branch: {"cca_ica": [...], "cca_eca": [...]} (cca_eca
    optional) — each a *full* continuous Series from the common carotid
    trunk through to that path's own tip (ICA or ECA), as stored in
    CaseState.segmentation_series, every point needing at least
    point_mm/tangent_mm/radius_mm (i.e. already segmented — unsegmented
    points are silently skipped, leaving a gap, so segment the whole path
    before building). The two paths deliberately overlap along the shared
    trunk — rasterizing both into the same mask just unions that overlap,
    which is exactly the bifurcation geometry.

    Returns (vtkPolyData or None, diag).
    """
    diag = {"branches_used": [b for b, s in series_by_branch.items() if s]}
    has_cca_eca = bool(series_by_branch.get("cca_eca"))
    n_expected = 3 if has_cca_eca else 2

    lo_mm, hi_mm = _series_bbox_mm(series_by_branch)
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64)
    full_shape = np.asarray(array_shape)

    # A fixed 10mm margin doesn't necessarily clear every disk's actual
    # extent (radius + half-thickness) — a point with a larger-than-usual
    # fitted radius could have its disk clipped flat by the crop boundary
    # itself rather than by a real anatomical end, which reads as the
    # model being "cut off" partway along, not at a true open end.
    margin = max(BBOX_MARGIN_MM, _max_radius_mm(series_by_branch) + DISK_HALF_THICKNESS_MM + 2.0)
    diag["bbox_margin_mm"] = margin
    crop_lo_vox = np.floor((lo_mm - margin - origin) / spacing).astype(int)
    crop_hi_vox = np.ceil((hi_mm + margin - origin) / spacing).astype(int)
    crop_lo_vox = np.maximum(crop_lo_vox, 0)
    crop_hi_vox = np.minimum(crop_hi_vox, full_shape)
    if np.any(crop_hi_vox <= crop_lo_vox):
        diag["error"] = "degenerate bounding box for segmented series"
        return None, diag

    crop_shape = tuple((crop_hi_vox - crop_lo_vox).tolist())
    crop_origin_mm = origin + crop_lo_vox * spacing
    mask = np.zeros(crop_shape, dtype=bool)

    # Trim each path's own ends *before* rasterizing — see _inset_series
    # for why this replaces the earlier approach of rasterizing the full
    # path and clipping the resulting surface afterward with an infinite
    # plane (a real failure mode on a tortuous vessel: the plane can also
    # slice a distant, unrelated loop of the same curving vessel, and
    # discard a real connected stretch of it instead of just the small
    # end cap). The inlet trim applies to the start of every path present
    # (they share the same trunk origin); each path's own distal trim
    # applies only to its own end.
    trimmed_by_branch = {}
    for branch in ("cca_ica", "cca_eca"):
        series = series_by_branch.get(branch) or []
        if len(series) >= 2:
            series = _inset_series(series, CUT_INSET_MM, at_start=True)
        if len(series) >= 2:
            series = _inset_series(series, CUT_INSET_MM, at_start=False)
        trimmed_by_branch[branch] = series

    for branch, series in trimmed_by_branch.items():
        if not series:
            continue
        n_stamped, n_substituted = _rasterize_branch(mask, crop_origin_mm, spacing, series)
        diag[f"{branch}_points_stamped"] = n_stamped
        diag[f"{branch}_points_substituted_outlier"] = n_substituted

    # The two branches are only connected in the first place by their
    # disks naturally overlapping near the shared bifurcation — but each
    # branch's own *start* just got trimmed away from exactly that area
    # (to give the inlet cut a clean full-diameter place to cut, same as
    # every other trim here), and a thin, path-local bridge (_bridge_gap
    # above) only ever bridges a branch to *itself*. CONFIRMED (on a real
    # patient's ECA) that once disks are no longer thickened past a real
    # vessel's own radius, that's not automatically enough margin to
    # still reach the other branch's own trimmed start too — the ECA
    # rasterized as its own fully disconnected island, 437 voxels lost to
    # _largest_component, not merely thin. Explicitly bridging each
    # secondary branch's own trimmed start to the *nearest point on the
    # inlet branch* (not necessarily its own start — the nearest point
    # along it, wherever that falls) closes exactly that gap without
    # reintroducing an oversized disk anywhere.
    inlet_branch_series = trimmed_by_branch.get("cca_ica") or trimmed_by_branch.get("cca_eca")
    if inlet_branch_series:
        inlet_pts = np.asarray([p["point_mm"] for p in inlet_branch_series], dtype=np.float64)
        for branch, series in trimmed_by_branch.items():
            if not series or series is inlet_branch_series:
                continue
            start = series[0]
            d = np.linalg.norm(inlet_pts - np.asarray(start["point_mm"]), axis=1)
            nearest = inlet_branch_series[int(np.argmin(d))]
            for sub in _bridge_gap(nearest, nearest["contour_mm"], start, start["contour_mm"]):
                _stamp_contour(mask, crop_origin_mm, spacing, sub["point_mm"], sub["tangent_mm"], sub["contour_mm"])

    if not mask.any():
        diag["error"] = "no voxels rasterized — no segmented points with a radius"
        return None, diag

    kept_volume = np.pad(mask.astype(np.uint8), 1)
    pad_origin_mm = crop_origin_mm - spacing

    resampled, iso_spacing = _resample_isotropic(kept_volume, spacing)
    img = _numpy_to_vtk_image(resampled, np.asarray(iso_spacing), pad_origin_mm)

    mc = vtk.vtkDiscreteFlyingEdges3D()
    mc.SetInputData(img)
    mc.SetValue(0, 1)
    mc.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(mc.GetOutput())
    diag["raw_points"] = poly.GetNumberOfPoints()

    poly = _largest_component(poly)
    poly = _smooth(poly)
    poly = _largest_component(poly)

    # Open the true ends: marching cubes caps every end of the mask
    # (CONFIRMED — _inset_series trimming what gets rasterized controls
    # *where* the mask ends, but never leaves that end open on its own),
    # so each one still needs an explicit cut. See _local_open_end for
    # why this is a *locally bounded* cut rather than carotid_seg.
    # meshing._clip_away's infinite plane. The inlet is shared by every
    # path present (they all start at the same trunk origin), so cutting
    # it once, at either path's own trimmed start point, is enough; each
    # path's own distal outlet (its trimmed end point) needs its own cut.
    inlet_series = trimmed_by_branch.get("cca_ica") or trimmed_by_branch.get("cca_eca") or []
    cut_points_xyz = []
    if inlet_series:
        p = inlet_series[0]
        poly = _local_open_end(poly, p["point_mm"], p["tangent_mm"], _cut_radius_mm(p), at_start=True)
        cut_points_xyz.append(np.asarray(_xyz(p["point_mm"])))
    for branch in ("cca_ica", "cca_eca"):
        branch_series = trimmed_by_branch.get(branch) or []
        if branch_series:
            p = branch_series[-1]
            poly = _local_open_end(poly, p["point_mm"], p["tangent_mm"], _cut_radius_mm(p), at_start=False)
            cut_points_xyz.append(np.asarray(_xyz(p["point_mm"])))
    poly = _largest_component(poly)

    n_loops, n_nonmanifold = count_boundary_loops(poly)
    if n_loops > n_expected:
        poly = _close_spurious_holes(poly, cut_points_xyz)
        poly = _largest_component(poly)
        n_loops, n_nonmanifold = count_boundary_loops(poly)
        diag["spurious_holes_closed"] = True

    diag.update({
        "ok": True,
        "n_points": poly.GetNumberOfPoints(),
        "n_cells": poly.GetNumberOfCells(),
        "n_boundary_loops": n_loops,
        "n_nonmanifold_edges": n_nonmanifold,
        "clean_topology": (n_loops == n_expected and n_nonmanifold == 0),
    })
    return poly, diag
