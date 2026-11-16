"""Calibration report: operator contours vs automatic methods.

Reads a project segmented by hand (it is never modified) and, for every
manually drawn lumen or outer wall, runs the automatic methods on the same
MPR section and scores them against the operator.

    .venv/bin/python scripts/calibration_report.py "/path/to/projects/PATIENT" \
        --dicom-root /path/to/dicom --out reports/PATIENT

Writes sections.csv (one row per section and method), summary.json and
summary.md into --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--dicom-alias", help="container path prefix stored in the project, e.g. /data/dicom")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calcium-hu", type=float, default=600.0)
    args = parser.parse_args(argv)

    os.environ["CAROTID_DICOM_ROOT"] = args.dicom_root
    os.environ["CAROTID_PROJECTS_ROOT"] = str(args.project.resolve().parent)
    if args.dicom_alias:
        os.environ["CAROTID_DICOM_ROOT_ALIAS"] = args.dicom_alias
    sys.path.insert(0, str(ROOT))
    import logging
    logging.disable(logging.INFO)
    from server import calibration, cases, projects, reslice

    info = projects.open_project(projects.ProjectRequest(path=str(args.project.resolve())))
    case = cases.get(info["case_id"])
    case.project_dir = None  # read-only: no autosave into the reference project

    rows = []
    for (side, branch), series in sorted(case.segmentation_series.items()):
        for index, point in enumerate(series):
            manual_wall = (point.get("wall_analysis") or {}).get("status") == "manual"
            if point.get("method") != "manual" and not manual_wall:
                continue
            image, *_ = reslice.sample_oblique_slice(
                case.array, case.spacing_mm, case.origin_mm, point["point_mm"], point["tangent_mm"],
                size_px=calibration.SIZE_PX, px_spacing_mm=calibration.PX_MM,
                background_value=case.array_min,
                frame=(point["u_vec"], point["v_vec"]) if "u_vec" in point else None,
            )
            for row in calibration.calibrate_section(image, point, case.modality, args.calcium_hu):
                rows.append({"side": side, "branch": branch, "section": index + 1, **row})

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["side", "branch", "section", "target", "method", "failure", "dice", "area_reference_mm2",
              "area_candidate_mm2", "area_error_pct", "mean_boundary_mm", "hausdorff_mm"]
    with open(args.out / "sections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = calibration.summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ["| target:method | sections | failures | median Dice | median area error % | "
             "p90 abs area error % | median boundary mm | p90 Hausdorff mm |", "|" + "---|" * 8]
    fmt = lambda v, p=2: "—" if v is None else f"{v:.{p}f}"
    for key, s in summary.items():
        lines.append(f"| {key} | {s['sections']} | {s['failures']} | {fmt(s['median_dice'], 3)} | "
                     f"{fmt(s['median_area_error_pct'], 1)} | {fmt(s['p90_abs_area_error_pct'], 1)} | "
                     f"{fmt(s['median_mean_boundary_mm'])} | {fmt(s['p90_hausdorff_mm'])} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not rows:
        print("No manual contours found: draw lumen contours with «Ручной контур» "
              "(and outer walls with «Ручная внешняя стенка») first.")


if __name__ == "__main__":
    main()
