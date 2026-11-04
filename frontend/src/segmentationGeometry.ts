import * as THREE from "three";

export type SegmentationPoint = {
  point_mm: number[];
  tangent_mm: number[];
  contour_mm?: number[][];
  contour_px?: number[][];
  radius_mm?: number;
  method?: string;
};
export type DisplayFlips = Record<string, { x: boolean; y: boolean }>;

// Backend points are in [z,y,x] mm. Apply the same acquisition-axis
// reflections as the slice viewer, then put superior on scene Y.
export function scenePoint(
  point: number[],
  origin: number[],
  flips?: DisplayFlips,
) {
  return new THREE.Vector3(
    (point[2] - origin[2]) * (flips?.axial?.x ? -1 : 1),
    (point[0] - origin[0]) * (flips?.coronal?.y ? -1 : 1),
    (point[1] - origin[1]) * (flips?.axial?.y ? -1 : 1),
  );
}

export function resampleRing(
  points: THREE.Vector3[],
  count = 18,
): THREE.Vector3[] {
  if (
    points.length < 3 ||
    count < 3 ||
    points.some((p) => !Number.isFinite(p.x + p.y + p.z))
  )
    return [];
  const lengths = points.map((p, i) =>
    p.distanceTo(points[(i + 1) % points.length]),
  );
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total < 1e-9) return [];
  let segment = 0,
    walked = 0;
  return Array.from({ length: count }, (_, i) => {
    const target = (total * i) / count;
    while (segment < points.length - 1 && walked + lengths[segment] <= target)
      walked += lengths[segment++];
    return points[segment]
      .clone()
      .lerp(
        points[(segment + 1) % points.length],
        lengths[segment] > 0 ? (target - walked) / lengths[segment] : 0,
      );
  });
}

export function alignRing(ring: THREE.Vector3[], reference: THREE.Vector3[]) {
  let best = ring,
    bestScore = Infinity;
  for (const candidate of [ring, [...ring].reverse()]) {
    for (let shift = 0; shift < ring.length; shift++) {
      const shifted = candidate.map(
        (_, i) => candidate[(i + shift) % ring.length],
      );
      const score = shifted.reduce(
        (sum, p, i) => sum + p.distanceToSquared(reference[i]),
        0,
      );
      if (score < bestScore) {
        best = shifted;
        bestScore = score;
      }
    }
  }
  return best;
}

// Match the native preview: one merged ribbon mesh; never bridge an
// unsegmented section. This preview is independent of the final CFD mesh.
export function wallGeometry(
  points: SegmentationPoint[],
  convert: (p: number[]) => THREE.Vector3,
) {
  const rings = points.map((p) =>
    resampleRing((p.contour_mm || []).map(convert)),
  );
  const vertices: number[] = [],
    indices: number[] = [];
  let segments = 0;
  for (let i = 0; i < rings.length - 1; i++) {
    const a = rings[i];
    if (!a.length || !rings[i + 1].length) continue;
    const b = alignRing(rings[i + 1], a),
      n = a.length,
      base = vertices.length / 3;
    for (const p of [...a, ...b]) vertices.push(p.x, p.y, p.z);
    for (let k = 0; k < n; k++) {
      const next = (k + 1) % n;
      indices.push(
        base + k,
        base + n + k,
        base + next,
        base + next,
        base + n + k,
        base + n + next,
      );
    }
    segments++;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3),
  );
  geometry.setIndex(indices);
  if (indices.length) geometry.computeVertexNormals();
  return { geometry, segments };
}

export function selectedRing(
  point: SegmentationPoint | undefined,
  radius: number,
  convert: (p: number[]) => THREE.Vector3,
) {
  if (!point) return [];
  const accepted = resampleRing((point.contour_mm || []).map(convert), 40);
  if (accepted.length) return accepted;
  const center = convert(point.point_mm);
  const tangent = new THREE.Vector3(...point.tangent_mm);
  if (tangent.lengthSq() < 1e-12) return [];
  tangent.normalize();
  const seed =
    Math.abs(tangent.z) < 0.9
      ? new THREE.Vector3(0, 0, 1)
      : new THREE.Vector3(0, 1, 0);
  const u = seed.addScaledVector(tangent, -seed.dot(tangent)).normalize();
  const v = tangent.clone().cross(u);
  return Array.from({ length: 40 }, (_, i) => {
    const theta = (i * 2 * Math.PI) / 40;
    const offset = u
      .clone()
      .multiplyScalar(Math.cos(theta) * radius)
      .addScaledVector(v, Math.sin(theta) * radius);
    return convert(
      point.point_mm.map((value, axis) => value + offset.getComponent(axis)),
    );
  }).filter((p) => Number.isFinite(p.distanceTo(center)));
}
