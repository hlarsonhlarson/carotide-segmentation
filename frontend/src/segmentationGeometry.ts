import * as THREE from "three";

export type PlaqueComposition = {
  plaque_area_mm2: number;
  calcified_area_mm2: number;
  lipid_area_mm2: number;
  fibrous_area_mm2: number;
  excluded_low_hu_area_mm2: number;
  contrast_partial_volume_area_mm2: number;
  calcium_hu_effective: number;
  lumen_reference_hu: number | null;
  calcium_in_lumen_area_mm2: number;
  calcified_pixels: number[][];
  lipid_pixels: number[][];
  fibrous_pixels: number[][];
  parameters: { calcium_hu: number; lipid_hu: number; fat_hu: number; fibrous_max_hu: number; edge_margin_mm: number };
};
export type SegmentationPoint = {
  point_mm: number[];
  tangent_mm: number[];
  contour_mm?: number[][];
  contour_px?: number[][];
  radius_mm?: number;
  method?: string;
  wall_analysis?: {
    wall_contour_px: number[][];
    high_density_pixels: number[][];
    annulus_area_mm2: number;
    high_density_area_mm2: number;
    max_radial_thickness_mm: number | null;
    edge_support_fraction: number | null;
    status: "proposal" | "manual";
    parameters: { calcium_hu: number };
    composition?: PlaqueComposition;
  };
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
export function wallRings(
  points: SegmentationPoint[],
  convert: (p: number[]) => THREE.Vector3,
) {
  return points.map((p) => resampleRing((p.contour_mm || []).map(convert)));
}

export function wallGeometry(
  points: SegmentationPoint[],
  convert: (p: number[]) => THREE.Vector3,
  // Interval i joins sections i and i+1; omit to draw every interval.
  includeInterval?: (i: number) => boolean,
  // Rings from wallRings(), when one series is drawn in several parts.
  precomputed?: THREE.Vector3[][],
) {
  const rings = precomputed ?? wallRings(points, convert);
  const vertices: number[] = [],
    indices: number[] = [];
  let segments = 0;
  for (let i = 0; i < rings.length - 1; i++) {
    const a = rings[i];
    if (!a.length || !rings[i + 1].length) continue;
    if (includeInterval && !includeInterval(i)) continue;
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

export const segmentationKey = (base: string, side: string, branch: string) =>
  `${base}/${side}/${branch}`;
export type SegmentationChange = (
  base: string,
  side: string,
  branch: string,
  points: SegmentationPoint[],
) => void;

export type VesselSegment = "cca" | "bifurcation" | "ica" | "eca";
export const vesselSegments: VesselSegment[] = ["cca", "bifurcation", "ica", "eca"];
export const vesselSegmentNames: Record<VesselSegment, string> = {
  cca: "ОСА", bifurcation: "Бифуркация", ica: "ВСА", eca: "НСА",
};

const SHARED_TOLERANCE_MM = 1.0;
const SHARED_RADIUS_FRACTION = 0.5;
const MAX_EXCURSION = 2;
export type VesselLabels = {
  cca_ica: VesselSegment[];
  cca_eca: VesselSegment[];
  split: boolean;
  // Leading sections that lie on the other branch's path (the common CCA).
  shared: { cca_ica: number; cca_eca: number };
};

function arcLengths(points: SegmentationPoint[]) {
  const arc = [0];
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1].point_mm, points[i].point_mm];
    arc.push(arc[i - 1] + Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  return arc;
}

function distanceToPolyline(p: number[], line: SegmentationPoint[]) {
  let best = Infinity;
  for (let i = 0; i < line.length; i++) {
    const a = line[i].point_mm;
    const b = line[Math.min(i + 1, line.length - 1)].point_mm;
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    const len2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
    const t = len2 > 0 ? Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2)) : 0;
    best = Math.min(best, Math.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]));
  }
  return best;
}

/**
 * Label every section of both branch series as CCA, bifurcation, ICA or ECA.
 *
 * Both series start in the CCA. Their shared run (sections within half the
 * lumen radius, at least 1 mm, of the other branch's path, allowing brief
 * excursions) is the CCA; the split is where they part. Sections
 * within `bifurcationMm` of the split along each path form the bifurcation
 * zone; distal sections belong to that branch's own artery. With only one
 * branch the split is unknown, so its sections keep the branch's label.
 * This is geometric bookkeeping along the operator's paths, not anatomy
 * detected from the image.
 */
export function labelVesselSegments(
  ica: SegmentationPoint[],
  eca: SegmentationPoint[],
  bifurcationMm = 10,
): VesselLabels {
  const own = (n: number, label: VesselSegment) => Array<VesselSegment>(n).fill(label);
  const unknown = { cca_ica: own(ica.length, "ica"), cca_eca: own(eca.length, "eca"), split: false,
                    shared: { cca_ica: 0, cca_eca: 0 } };
  if (!ica.length || !eca.length) return unknown;
  const label = (series: SegmentationPoint[], other: SegmentationPoint[], distal: VesselSegment) => {
    // Two independently traced paths wander inside the same CCA lumen, so
    // the tolerance scales with the section radius, and a short excursion
    // (up to MAX_EXCURSION sections) does not end the common run.
    let shared = 0, outside = 0;
    for (let i = 0; i < series.length; i++) {
      const tolerance = Math.max(SHARED_TOLERANCE_MM,
        SHARED_RADIUS_FRACTION * (series[i].radius_mm ?? 0));
      if (distanceToPolyline(series[i].point_mm, other) <= tolerance) {
        shared = i + 1;
        outside = 0;
      } else if (++outside > MAX_EXCURSION) break;
    }
    if (shared === 0 || shared === series.length) return null;
    const arc = arcLengths(series);
    const split = arc[shared - 1];
    return {
      shared,
      labels: series.map((_, i): VesselSegment => {
        if (Math.abs(arc[i] - split) <= bifurcationMm) return "bifurcation";
        return i < shared ? "cca" : distal;
      }),
    };
  };
  const a = label(ica, eca, "ica"), b = label(eca, ica, "eca");
  if (!a || !b) return unknown;
  return { cca_ica: a.labels, cca_eca: b.labels, split: true,
           shared: { cca_ica: a.shared, cca_eca: b.shared } };
}
