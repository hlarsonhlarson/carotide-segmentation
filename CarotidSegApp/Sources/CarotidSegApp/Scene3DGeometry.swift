import SceneKit

/// Shared SCNVector3 math and wall-ribbon geometry used by every 3D
/// preview in the app (the main path view and the segmentation-time wall
/// preview) — factored out so both build the exact same lofted-wall shape
/// from the exact same contour data instead of drifting apart.

func scn_dist(_ a: SCNVector3, _ b: SCNVector3) -> CGFloat {
    let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z
    return sqrt(dx * dx + dy * dy + dz * dz)
}

func scn_distSquared(_ a: SCNVector3, _ b: SCNVector3) -> CGFloat {
    let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz
}

func scn_lerp(_ a: SCNVector3, _ b: SCNVector3, _ t: CGFloat) -> SCNVector3 {
    SCNVector3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)
}

func scn_cross(_ a: SCNVector3, _ b: SCNVector3) -> SCNVector3 {
    SCNVector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

func scn_normalize(_ v: SCNVector3) -> SCNVector3 {
    let len = scn_dist(SCNVector3Zero, v)
    guard len > 0 else { return v }
    return SCNVector3(v.x / len, v.y / len, v.z / len)
}

/// world-mm [z, y, x] -> scene (x, z, y): the one axis remapping every 3D
/// view in this app uses, so a point built here lines up with one built
/// anywhere else.
func mmToScene(_ mm: [Double]) -> SCNVector3 {
    SCNVector3(Float(mm[2]), Float(mm[0]), Float(mm[1]))
}

/// Resamples a closed 3D contour loop to `count` evenly (by arc length)
/// spaced points. The raw contours from the server are boundary walks of
/// whatever pixel count the detected vessel wall happened to produce —
/// different at every path index — so a fixed-count, evenly-spaced
/// resampling is what gives consecutive rings a consistent number of
/// vertices to loft a ribbon between.
func resampleClosedLoopMM(_ points: [[Double]], count: Int) -> [SCNVector3] {
    guard points.count >= 3 else { return [] }
    let vecs = points.map(mmToScene)

    var cum: [CGFloat] = [0]
    for i in 1..<vecs.count {
        cum.append(cum[i - 1] + scn_dist(vecs[i - 1], vecs[i]))
    }
    let closingDist = scn_dist(vecs[vecs.count - 1], vecs[0])
    let total = cum[cum.count - 1] + closingDist
    guard total > 0 else { return [] }

    var result: [SCNVector3] = []
    result.reserveCapacity(count)
    for k in 0..<count {
        let target = total * CGFloat(k) / CGFloat(count)
        var idx = 0
        while idx < cum.count - 1 && cum[idx + 1] < target { idx += 1 }
        let segStart = cum[idx]
        let a = vecs[idx]
        let b: SCNVector3
        let segLen: CGFloat
        if idx == vecs.count - 1 {
            b = vecs[0]
            segLen = closingDist
        } else {
            b = vecs[idx + 1]
            segLen = cum[idx + 1] - cum[idx]
        }
        let t = segLen > 0 ? (target - segStart) / segLen : 0
        result.append(scn_lerp(a, b, t))
    }
    return result
}

/// Each contour is extracted independently from its own oblique slice, so
/// neither its starting point nor its winding direction is guaranteed to
/// line up with the next ring's — without this, a ribbon lofted between
/// them twists. Tries every rotation (and both windings) of `ring`
/// against `reference` and keeps whichever lines up corresponding points
/// most closely.
func bestAlignedRing(_ ring: [SCNVector3], to reference: [SCNVector3]) -> [SCNVector3] {
    let n = reference.count
    guard ring.count == n, n > 0 else { return ring }
    var bestShift = 0
    var bestReversed = false
    var bestScore = CGFloat.greatestFiniteMagnitude
    for reversed in [false, true] {
        let candidate = reversed ? Array(ring.reversed()) : ring
        for shift in 0..<n {
            var score: CGFloat = 0
            for k in 0..<n {
                score += scn_distSquared(reference[k], candidate[(k + shift) % n])
            }
            if score < bestScore {
                bestScore = score
                bestShift = shift
                bestReversed = reversed
            }
        }
    }
    let candidate = bestReversed ? Array(ring.reversed()) : ring
    return (0..<n).map { candidate[($0 + bestShift) % n] }
}

func lineNode(points: [SCNVector3], color: NSColor) -> SCNNode {
    let source = SCNGeometrySource(vertices: points)
    var indices: [Int32] = []
    for i in 0..<(points.count - 1) {
        indices.append(Int32(i))
        indices.append(Int32(i + 1))
    }
    let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
    let element = SCNGeometryElement(
        data: indexData, primitiveType: .line,
        primitiveCount: indices.count / 2, bytesPerIndex: MemoryLayout<Int32>.size
    )
    let geometry = SCNGeometry(sources: [source], elements: [element])
    let material = SCNMaterial()
    material.diffuse.contents = color
    material.lightingModel = .constant
    geometry.materials = [material]
    return SCNNode(geometry: geometry)
}

/// A closed loop through `points` (connects the last point back to the
/// first) — e.g. for drawing a ring's own outline, as opposed to
/// lineNode's open polyline.
func closedLoopNode(points: [SCNVector3], color: NSColor) -> SCNNode {
    guard let first = points.first, points.count > 1 else { return SCNNode() }
    return lineNode(points: points + [first], color: color)
}

/// Builds one merged mesh for every ribbon segment lofted between
/// consecutive accepted contours in `contoursByIndex` (world-mm per
/// index) — the shared "growing wall" geometry both the live
/// segmentation preview and the main path view render, muted to `color`.
/// Merged into a single SCNGeometry (one draw call) rather than one node
/// per segment — with dozens of points along a branch this was the
/// difference between smooth and visibly laggy camera dragging, since
/// SceneKit has to sort and blend every translucent node's triangles
/// against every other one on screen each frame.
func buildWallMesh(contoursByIndex: [Int: [[Double]]], color: NSColor, ringResolution: Int = 18) -> SCNNode? {
    let sortedIndices = contoursByIndex.keys.sorted()
    var ringsByIndex: [Int: [SCNVector3]] = [:]
    for idx in sortedIndices {
        guard let raw = contoursByIndex[idx] else { continue }
        let ring = resampleClosedLoopMM(raw, count: ringResolution)
        guard !ring.isEmpty else { continue }
        ringsByIndex[idx] = ring
    }

    var vertices: [SCNVector3] = []
    var indices: [Int32] = []
    for idx in sortedIndices {
        guard let nextIdx = sortedIndices.first(where: { $0 == idx + 1 }),
              let ringA = ringsByIndex[idx] else { continue }
        guard var ringB = ringsByIndex[nextIdx] else { continue }
        ringB = bestAlignedRing(ringB, to: ringA)
        let n = ringA.count
        let base = Int32(vertices.count)
        vertices.append(contentsOf: ringA)
        vertices.append(contentsOf: ringB)
        for k in 0..<n {
            let a0 = base + Int32(k)
            let a1 = base + Int32((k + 1) % n)
            let b0 = base + Int32(n + k)
            let b1 = base + Int32(n + (k + 1) % n)
            indices.append(contentsOf: [a0, b0, a1])
            indices.append(contentsOf: [a1, b0, b1])
        }
    }
    guard !indices.isEmpty else { return nil }

    let source = SCNGeometrySource(vertices: vertices)
    let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
    let element = SCNGeometryElement(
        data: indexData, primitiveType: .triangles,
        primitiveCount: indices.count / 3, bytesPerIndex: MemoryLayout<Int32>.size
    )
    let geometry = SCNGeometry(sources: [source], elements: [element])
    let material = SCNMaterial()
    material.diffuse.contents = color
    material.lightingModel = .physicallyBased
    material.isDoubleSided = true
    material.transparency = 0.5
    // Translucent preview geometry shouldn't fight the depth buffer with
    // the opaque line/sphere markers also in the scene — cheaper to
    // render and avoids depth-fighting artifacts between branches.
    material.writesToDepthBuffer = false
    geometry.materials = [material]
    return SCNNode(geometry: geometry)
}

/// A ring perpendicular to the centerline's tangent at a given point —
/// used both for the "current slice level" indicator during segmentation
/// and could be reused anywhere else a cross-section marker is needed.
/// Built via Gram-Schmidt from the tangent (same approach server/
/// reslice.py uses to build the oblique sampling plane itself).
func crossSectionRingNode(centerMM: [Double], tangentMM: [Double], radiusMM: Double, color: NSColor) -> SCNNode {
    let center = mmToScene(centerMM)
    var tangent = mmToScene(tangentMM)
    let tangentLen = scn_dist(SCNVector3Zero, tangent)
    guard tangentLen > 0 else { return SCNNode() }
    tangent = SCNVector3(tangent.x / tangentLen, tangent.y / tangentLen, tangent.z / tangentLen)

    let arbitrary = abs(tangent.y) > 0.9 ? SCNVector3(1, 0, 0) : SCNVector3(0, 1, 0)
    let u = scn_normalize(scn_cross(tangent, arbitrary))
    let v = scn_cross(tangent, u)

    let radius = CGFloat(radiusMM)
    let segments = 40
    var points: [SCNVector3] = []
    for i in 0...segments {
        let theta = Float(i) / Float(segments) * 2 * Float.pi
        let c = CGFloat(cosf(theta)), s = CGFloat(sinf(theta))
        points.append(SCNVector3(
            center.x + c * radius * u.x + s * radius * v.x,
            center.y + c * radius * u.y + s * radius * v.y,
            center.z + c * radius * u.z + s * radius * v.z
        ))
    }
    let node = lineNode(points: points, color: color)
    node.renderingOrder = 10
    return node
}
