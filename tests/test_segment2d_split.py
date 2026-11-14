import numpy as np

from server import segment2d

CENTER = (79.5, 79.5)
MAX_R = 24.0


def _image(*discs, background=40.0, lumen=350.0):
    yy, xx = np.mgrid[0:160, 0:160]
    image = np.full((160, 160), background, dtype=np.float32)
    for cy, cx, r in discs:
        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = lumen
    return image


def _radius_px(contour):
    c = np.asarray(contour)
    area = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) - np.dot(c[:, 1], np.roll(c[:, 0], 1)))
    # Offset from the true lumen centre (80, 80), not the reslice centre.
    return float(np.sqrt(area / np.pi)), float(np.hypot(*(c.mean(0) - 80.0)))


def test_isolated_lumen_is_unchanged():
    r, offset = _radius_px(segment2d.threshold_contour(_image((80, 80, 7)), CENTER, "CT", MAX_R))
    assert abs(r - 7) < 0.6 and offset < 0.5


def test_touching_vessel_with_wide_contact_is_excluded():
    # Neighbour overlaps the lumen: contact chord is ~60% of lumen diameter.
    image = _image((80, 80, 7), (80, 92, 8))
    r, offset = _radius_px(segment2d.threshold_contour(image, CENTER, "CT", MAX_R))
    assert r < 8.0 and offset < 1.5


def test_thin_bridge_is_cut():
    image = _image((80, 80, 7), (80, 100, 7))
    image[79:82, 80:100] = 350.0
    r, offset = _radius_px(segment2d.threshold_contour(image, CENTER, "CT", MAX_R))
    # Growing the core back may keep a stub of the bridge no longer than the
    # cut radius (~4 px here); without the cut the far disc doubles the area.
    assert r < 7.5 and offset < 1.6


def test_elongated_lumen_is_not_split():
    yy, xx = np.mgrid[0:160, 0:160]
    image = np.full((160, 160), 40.0, dtype=np.float32)
    image[((yy - 80) / 6.0) ** 2 + ((xx - 80) / 11.0) ** 2 <= 1] = 350.0
    r, offset = _radius_px(segment2d.threshold_contour(image, CENTER, "CT", MAX_R))
    assert abs(r - np.sqrt(6 * 11)) < 0.7 and offset < 0.5
