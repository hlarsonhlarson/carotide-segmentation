"""Prepare a synthetic project through the same HTTP API used by the browser."""
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def request(path, data=None):
    req = Request('http://127.0.0.1:8000' + path,
                  data=None if data is None else json.dumps(data).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=180) as response:
        return json.load(response)


def main():
    folder = Path('/data/projects/demo')
    if (folder / 'project.json').exists():
        print('Existing demo preserved; edits were not overwritten.')
        return
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit('Demo folder is not empty; refusing to overwrite it')
    source = '/data/dicom/synthetic-demo'
    series = request('/series?path=' + source)['series']
    case = request('/cases', {'path': source, 'series_uid': series[0]['series_uid']})
    base = '/cases/' + case['case_id']
    branch = base + '/segment/right/cca_ica'
    for z in (8, 18, 28):
        request(base + '/centerline/right/cca_ica/points', {'point': [z, 16, 16]})
    prepared = request(branch + '/prepare', {})
    for i in range(prepared['n_points']):
        request(branch + '/point/' + str(i), {'method': 'circle', 'radius_mm': 3})
    request(base + '/segment/right/build_surface', {})
    request(base + '/segment/right/su2_mesh', {})
    for _ in range(180):
        status = request(base + '/segment/right/su2_mesh/status')
        if status['state'] == 'done':
            break
        if status['state'] != 'running':
            raise RuntimeError(status)
        time.sleep(1)
    else:
        raise TimeoutError('Demo mesh did not finish within 180 seconds')
    # Publish the project only after geometry is ready.
    request(base + '/project', {'path': str(folder)})
    request(base + '/save/project', {})
    print('Synthetic demo saved at /data/projects/demo')


if __name__ == '__main__':
    main()
