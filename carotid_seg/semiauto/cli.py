"""Run the semi-automatic lumen segmentation on one CTA.

Points are physical millimetres in the image frame, not voxel indices and not
cutting planes. Example::

    python -m carotid_seg.semiauto.cli \\
        --image /path/to/cta.nii.gz \\
        --cca 12.0 -140.0 80.0 \\
        --ica 18.0 -20.0 160.0 \\
        --out /path/to/review

A DICOM series directory is accepted as ``--image``. The run writes the lumen,
CCA/ICA/ECA labels, an uncertainty mask and a JSON report in the original
geometry. It does not report a clinical accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .params import LumenParams
from .segment import review_slice, segment_lumen


def read_image(path: Path) -> sitk.Image:
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(path))
        if not files:
            raise SystemExit(f"В папке нет DICOM-серии: {path}")
        reader.SetFileNames(files)
        return reader.Execute()
    return sitk.ReadImage(str(path))


def _point(values):
    if len(values) != 3:
        raise SystemExit("Точка задаётся тремя координатами x y z в миллиметрах.")
    return tuple(float(v) for v in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Полуавтоматический просвет ОСА/ВСА/НСА на КТА")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--cca", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--ica", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--eca", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vesselness", action="store_true",
                        help="Только пометить сомнительную границу. Маску не меняет.")
    args = parser.parse_args(argv)

    image = read_image(args.image)
    params = LumenParams(use_vesselness=args.vesselness)
    result = segment_lumen(image, _point(args.cca), _point(args.ica), None if args.eca is None else _point(args.eca), params)
    args.out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result.lumen, str(args.out / "lumen.nii.gz"))
    sitk.WriteImage(result.labels, str(args.out / "labels.nii.gz"))
    sitk.WriteImage(result.uncertainty, str(args.out / "uncertainty.nii.gz"))
    sitk.WriteImage(result.component, str(args.out / "component.nii.gz"))

    labels = sitk.GetArrayFromImage(result.labels)
    present = np.argwhere(labels > 0)
    if len(present):
        mid = present[len(present) // 2]
        rgb = review_slice(result, image, axis=0, index=int(mid[0]))
        sitk.WriteImage(sitk.GetImageFromArray(rgb, isVector=True), str(args.out / "overlay_k.png"))

    payload = {
        "status": result.status,
        "requests": result.requests,
        "qa": _jsonable(result.qa),
        "labels": {"1": "CCA", "2": "ICA", "3": "ECA"},
        "uncertainty_bits": {
            "1": "boundary", "2": "bloom", "4": "gap", "8": "canal",
            "16": "eca_ambiguous", "32": "unlabeled_extension",
            "64": "subocclusion", "128": "fov", "256": "volume_cap",
        },
        "not_a_clinical_validation": True,
    }
    (args.out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(result.status)
    for request in result.requests:
        print(request)
    return 0 if result.status in ("ok", "needs_review", "gap") else 1


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
