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


def test_frame_matches_wall_coordinates_and_units(tmp_path):
    data = extract_wall_time_series(tmp_path, fixtures(tmp_path), .01)
    np.testing.assert_allclose(data["points_mm"][0], [100, 0, 0])
    np.testing.assert_allclose(data["times_s"], [.1])
    np.testing.assert_allclose(data["wss_vec_pa"][0, 0], [2, 0, 0])


@pytest.mark.parametrize("option, message", [
    ("reorder", "node order"), ("missing_pressure", "missing skin friction or pressure"),
    ("duplicate_ids", "duplicate GlobalNodeID"),
])
def test_bad_output_cannot_be_mapped_silently(tmp_path, option, message):
    with pytest.raises(ValueError, match=message):
        extract_wall_time_series(tmp_path, fixtures(tmp_path, **{option: True}), .01)
