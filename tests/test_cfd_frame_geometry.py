import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from server.su2_postprocess import extract_wall_time_series


def array(values, name):
    result = numpy_to_vtk(np.asarray(values), deep=True)
    result.SetName(name)
    return result


def fixtures(tmp_path, reorder=False, missing_pressure=False, duplicate_ids=False):
    xyz_mm = np.array([[100., 0., 0.], [100., 1., 0.], [100., 0., 1.]])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz_mm, deep=True))
    triangle = vtk.vtkTriangle()
    for i in range(3):
        triangle.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(triangle)
    surf = vtk.vtkPolyData()
    surf.SetPoints(points)
    surf.SetPolys(cells)
    surf.GetPointData().AddArray(array([1, 1 if duplicate_ids else 2, 3], "GlobalNodeID"))
    surf.GetCellData().AddArray(array([1], "ModelFaceID"))
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(surf)
    writer.SetFileName(str(tmp_path / "wall.vtp"))
    writer.Write()
    flow_points = vtk.vtkPoints()
    flow_points.SetData(numpy_to_vtk(xyz_mm[[1, 0, 2] if reorder else [0, 1, 2]] / 1000, deep=True))
    flow = vtk.vtkUnstructuredGrid()
    flow.SetPoints(flow_points)
    flow.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
    flow.GetPointData().AddArray(array(np.tile([4., 0., 0.], (3, 1)), "Skin_Friction_Coefficient"))
    if not missing_pressure:
        flow.GetPointData().AddArray(array([10000., 10001., 10002.], "Pressure"))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(flow)
    writer.SetFileName(str(tmp_path / "flow_10.vtu"))
    writer.Write()
    return (tmp_path / "wall.vtp").read_bytes()


APP_CFG = "INC_NONDIM= DIMENSIONAL\nINC_DENSITY_INIT= 1050.0\nINC_VELOCITY_INIT= (0.0, 0.0, 0.0)\n"


def test_frame_matches_wall_coordinates_and_units(tmp_path):
    (tmp_path / "run.cfg").write_text(APP_CFG)
    data = extract_wall_time_series(tmp_path, fixtures(tmp_path), .01)
    np.testing.assert_allclose(data["points_mm"][0], [100, 0, 0])
    np.testing.assert_allclose(data["times_s"], [.1])
    # Zero initial velocity: SU2's reference dynamic pressure is 1, Cf is WSS in Pa.
    np.testing.assert_allclose(data["wss_vec_pa"][0, 0], [4, 0, 0])


def test_skin_friction_scale_follows_su2_reference_values(tmp_path):
    from server.su2_postprocess import skin_friction_scale_pa
    (tmp_path / "run.cfg").write_text(APP_CFG)
    assert skin_friction_scale_pa(tmp_path) == 1.0
    (tmp_path / "run.cfg").write_text(APP_CFG.replace("(0.0, 0.0, 0.0)", "(0.0, 0.0, 0.2)"))
    assert np.isclose(skin_friction_scale_pa(tmp_path), 0.5 * 1050 * 0.04)
    (tmp_path / "run.cfg").write_text("INC_NONDIM= REFERENCE_VALUES\nINC_DENSITY_REF= 2\nINC_VELOCITY_REF= 3\n")
    assert np.isclose(skin_friction_scale_pa(tmp_path), 9.0)


@pytest.mark.parametrize("option, message", [
    ("reorder", "node order"), ("missing_pressure", "missing skin friction or pressure"),
    ("duplicate_ids", "duplicate GlobalNodeID"),
])
def test_bad_output_cannot_be_mapped_silently(tmp_path, option, message):
    with pytest.raises(ValueError, match=message):
        extract_wall_time_series(tmp_path, fixtures(tmp_path, **{option: True}), .01)


def test_pinched_boundary_is_split_into_simple_cycles():
    from server.su2_export import _simple_cycles
    # Figure-of-eight: a 5-cycle and a 3-cycle sharing vertex 0.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (5, 6), (6, 0)]
    cycles = sorted(_simple_cycles(edges), key=len, reverse=True)
    assert [len(c) for c in cycles] == [5, 3]
    assert set(cycles[0]) == {0, 1, 2, 3, 4} and set(cycles[1]) == {0, 5, 6}
    ring = [(i, (i + 1) % 6) for i in range(6)]
    assert [len(c) for c in _simple_cycles(ring)] == [6]


@pytest.mark.parametrize("clockwise", [False, True])
def test_disc_cap_is_watertight_and_well_shaped(clockwise):
    from vtk.util.numpy_support import vtk_to_numpy
    from server.su2_export import _disc_cap
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False) * (-1 if clockwise else 1)
    loop = np.column_stack([3 * np.cos(t), 3 * np.sin(t), 0.2 * np.sin(3 * t)])  # wavy rim
    cap = _disc_cap(loop, 0.4)
    assert cap is not None
    pts = vtk_to_numpy(cap.GetPoints().GetData())
    faces = vtk_to_numpy(cap.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    np.testing.assert_allclose(pts[:48], loop)  # rim untouched, so it merges with the wall
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) == {1, 2} and (counts == 1).sum() == 48  # only the rim is open
    p = pts[faces]
    a, b, c = (np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3))
    s = (a + b + c) / 2
    quality = 4 * np.sqrt(3) * np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0)) / (a * a + b * b + c * c)
    assert quality.min() > 0.3
