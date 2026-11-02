"""Import a legacy project's annotations into a NEW folder; never alter the source."""
import argparse
import json
from pathlib import Path

from . import cases, projects, workflow


def migrate(source, destination):
    from .app import _load_dicom_case, SU2ParamsRequest
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or destination.exists():
        raise ValueError('Destination must be a new folder outside the source project')
    manifest = json.loads((source / 'project.json').read_text())
    if 'format_version' in manifest:
        raise ValueError('Only unversioned legacy projects need migration')
    case, _ = _load_dicom_case(manifest['source_path'], manifest.get('series_uid'))
    try:
        for side in workflow.SIDES:
            path = source / f'{side}_centerline.json'
            if path.exists():
                for branch, values in json.loads(path.read_text()).items():
                    if branch not in workflow.BRANCHES:
                        continue
                    case.control_points[side, branch] = values.get('control_points', [])
                    case.centerline_points[side, branch] = values.get('centerline_points', [])
            path = source / f'{side}_segmentation.json'
            if path.exists():
                for branch, values in json.loads(path.read_text()).items():
                    if branch in workflow.BRANCHES:
                        case.segmentation_series[side, branch] = values
            path = source / f'{side}_su2_params.json'
            if path.exists():
                case.su2_params[side] = SU2ParamsRequest(**json.loads(path.read_text())).model_dump()
            case.stale[side].update(workflow.STAGES)
        for side in workflow.SIDES:
            for branch in workflow.BRANCHES:
                projects.validate_branch({
                    'control_points': case.control_points.get((side, branch), []),
                    'centerline_points': case.centerline_points.get((side, branch), []),
                    'segmentation': case.segmentation_series.get((side, branch), []),
                }, case.array.shape)
        destination.mkdir(parents=True)
        case.project_dir = str(destination)
        case.project_warnings = ['Imported legacy annotations: review paths and contours against the images before rebuilding.']
        data = projects.snapshot(case)
        data['migration'] = {'source': str(source), 'review_required': True}
        projects.atomic_write(destination / 'project.json', json.dumps(data, allow_nan=False, indent=2).encode())
        return destination
    finally:
        cases._CASES.pop(case.case_id, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    try:
        print(migrate(args.source, args.destination))
        print('Review all imported paths and contours against the images. Rebuild geometry and CFD; old results remain in the original folder.')
    except (OSError, ValueError, KeyError) as e:
        parser.exit(1, f'Migration failed: {e}\n')


if __name__ == '__main__':
    main()
