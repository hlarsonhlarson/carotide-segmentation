import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  alignRing,
  resampleRing,
  scenePoint,
  selectedRing,
  wallGeometry,
  type SegmentationPoint,
} from "./segmentationGeometry";
const point = (z: number, radius = 2): SegmentationPoint => ({
  point_mm: [z, 0, 0],
  tangent_mm: [1, 0, 0],
  contour_mm: Array.from({ length: 12 }, (_, i) => [
    z,
    Math.cos((i * Math.PI) / 6) * radius,
    Math.sin((i * Math.PI) / 6) * radius,
  ]),
});
const convert = (p: number[]) => scenePoint(p, [0, 0, 0]);

describe("segmentation wall preview", () => {
  it("overlays world-mm contour centres on voxel paths with nonzero origin and reversed Z", () => {
    const origin = [-120, 40, -80],
      spacing = [0.7, 0.52, 0.52],
      voxel = [136, 284, 186];
    const flips = {
      axial: { x: false, y: false },
      coronal: { x: false, y: true },
    };
    const path = scenePoint(
      voxel.map((v, i) => v * spacing[i]),
      [0, 0, 0],
      flips,
    );
    const worldCenter = voxel.map((v, i) => origin[i] + v * spacing[i]);
    const contourCenter = scenePoint(worldCenter, origin, flips);
    expect(path.distanceTo(contourCenter)).toBeLessThan(1e-10);
    const ring = selectedRing(
      { point_mm: worldCenter, tangent_mm: [1, 0, 0] },
      3,
      (p) => scenePoint(p, origin, flips),
    );
    const centre = ring
      .reduce((sum, p) => sum.add(p), new THREE.Vector3())
      .divideScalar(ring.length);
    expect(centre.distanceTo(path)).toBeLessThan(1e-10);
  });

  it("resamples closed rings with repeated endpoints and aligns reversed winding", () => {
    const source = point(0).contour_mm!.map(convert);
    const ring = resampleRing([...source, source[0]], 18);
    expect(ring).toHaveLength(18);
    const reversed = [...ring.slice(6), ...ring.slice(0, 6)].reverse();
    const aligned = alignRing(reversed, ring);
    expect(aligned.every((p, i) => p.distanceTo(ring[i]) < 1e-9)).toBe(true);
    expect(
      resampleRing([
        new THREE.Vector3(),
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]),
    ).toEqual([]);
  });
  it("builds one wall only between adjacent accepted contours and removes gaps", () => {
    const points = [
      point(0),
      point(1),
      { ...point(2), contour_mm: [] },
      point(3),
      point(4),
    ];
    const result = wallGeometry(points, convert);
    expect(result.segments).toBe(2);
    expect(result.geometry.getIndex()!.count).toBe(2 * 18 * 6);
    const positions = result.geometry.getAttribute("position");
    const indices = result.geometry.getIndex()!;
    for (let i = 0; i < indices.count; i += 3) {
      const heights = [0, 1, 2].map((k) => positions.getY(indices.getX(i + k)));
      expect(Math.max(...heights) - Math.min(...heights)).toBeLessThanOrEqual(
        1,
      );
    }
    result.geometry.dispose();
  });
  it("updates accepted contours even when the point counts stay unchanged", () => {
    const before = wallGeometry([point(0), point(1)], convert);
    const after = wallGeometry([point(0, 4), point(1, 4)], convert);
    before.geometry.computeBoundingBox();
    after.geometry.computeBoundingBox();
    expect(after.geometry.boundingBox!.max.x).toBeCloseTo(
      before.geometry.boundingBox!.max.x * 2,
    );
    before.geometry.dispose();
    after.geometry.dispose();
  });
  it("uses the accepted contour for selection; otherwise a tangent-normal radius ring", () => {
    const accepted = selectedRing(point(3), 12, convert);
    expect(accepted).toHaveLength(40);
    expect(Math.max(...accepted.map((p) => Math.hypot(p.x, p.z)))).toBeCloseTo(
      2,
    );
    const fallback = selectedRing(
      { point_mm: [3, 4, 5], tangent_mm: [1, 1, 0] },
      2.5,
      convert,
    );
    const center = convert([3, 4, 5]);
    const tangent = convert([1, 1, 0]).normalize();
    fallback.forEach((p) => {
      expect(p.distanceTo(center)).toBeCloseTo(2.5);
      expect(p.clone().sub(center).dot(tangent)).toBeCloseTo(0);
    });
  });
  it("keeps the descending CT orientation consistent with 2D and the main path viewer", () => {
    const p = scenePoint([31, 42, 53], [30, 40, 50], {
      axial: { x: false, y: false },
      coronal: { x: false, y: true },
    });
    expect(p.toArray()).toEqual([3, -1, 2]);
  });
});
