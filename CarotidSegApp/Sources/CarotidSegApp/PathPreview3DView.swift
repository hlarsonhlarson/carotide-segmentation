import SceneKit
import SwiftUI

/// Light box-average smoothing for display — the clicked+auto-stitched
/// path moves one voxel at a time (26-connectivity geodesic stitching),
/// which staircases visibly at 3D scale. Matches the same smoothing intent
/// as server/path_series.py's smooth_path (that one shapes the actual
/// segmentation tangents; this one is purely for how the curve looks here).
///
/// Smoothing is done *per segment* between consecutive control points,
/// never averaging across a control point into the next segment — doing
/// a plain global box-average and then snapping each control point back
/// to its exact position afterward (tried first) still looked wrong: the
/// window right next to a control point would blend in points from
/// *beyond* it, so the curve overshot to one side before snapping back,
/// making the point's sphere look like it was floating beside the line
/// rather than sitting on it, especially at a sharp turn like a
/// bifurcation. Keeping each segment's averaging window clamped to that
/// segment removes that cross-talk, and every control point is still
/// pinned to its exact raw position afterward as a final guarantee.
private func smoothPathPoints(_ points: [[Int]], controlPoints: [[Int]], windowRadius: Int = 3) -> [[Double]] {
    guard points.count > 4 else { return points.map { $0.map(Double.init) } }

    // Control points appear in the dense path unchanged and in the same
    // order (the server builds it by stitching them together one segment
    // at a time) — find each one's index, searching forward from the
    // previous match so repeated/nearby coordinates can't match out of
    // order.
    var boundaries: [Int] = []
    var searchFrom = 0
    for cp in controlPoints {
        if let idx = points[searchFrom...].firstIndex(where: { $0 == cp }) {
            boundaries.append(idx)
            searchFrom = idx
        }
    }
    if boundaries.first != 0 { boundaries.insert(0, at: 0) }
    if boundaries.last != points.count - 1 { boundaries.append(points.count - 1) }

    var result = points.map { $0.map(Double.init) }
    for s in 0..<(boundaries.count - 1) {
        let lo = boundaries[s]
        let hi = boundaries[s + 1]
        guard hi > lo else { continue }
        for i in lo...hi {
            var sum = [0.0, 0.0, 0.0]
            var count = 0.0
            for j in max(lo, i - windowRadius)...min(hi, i + windowRadius) {
                sum[0] += Double(points[j][0]); sum[1] += Double(points[j][1]); sum[2] += Double(points[j][2])
                count += 1
            }
            result[i] = [sum[0] / count, sum[1] / count, sum[2] / count]
        }
    }
    for idx in boundaries {
        result[idx] = points[idx].map(Double.init)
    }
    return result
}

/// Owns the camera's one-time initial framing and the drag handler for
/// right-drag-zoom/middle-drag-pan (see Interactive3DSceneView).
@MainActor
final class PathPreview3DCoordinator {
    var cameraNode: SCNNode?
    // Re-frame whenever this changes (a point was added/removed), not
    // just once ever — the user wants the camera to keep re-centering on
    // the growing path as they mark it, not have to hit Recenter by hand
    // after every click.
    var lastControlPointCount = -1
    // Re-frame on this changing too — confirmed, via a real seeded-
    // segmentation run, as the actual cause of the wall preview looking
    // like it "didn't transfer" into this window: auto-segmenting a path
    // adds dozens of wall-contour rings (each with a real physical
    // radius, much bigger than the thin path line) without adding any
    // new *control* point, so lastControlPointCount alone never changes
    // and the camera stayed at whatever tiny framing it had from before
    // any wall existed — the mesh was really being built and added to
    // the scene the whole time, just left outside the still-unmoved
    // camera's view.
    var lastWallContourCount = -1
    var lastSignature: Int?
    let dragHandler = OrbitCameraDragHandler()
}

/// A 3D preview of the traced path(s) for the currently-selected side — the
/// "how it actually travels through space" view that replaces trying to
/// show the whole centerline flattened onto every 2D slice. Also renders
/// each branch's preliminary vessel-wall preview (built while stepping
/// through segmentation in SegmentPathView) so it stays visible here —
/// "in the 4th window" — after that sheet is closed, fading in the first
/// time it appears.
struct PathPreview3DView: NSViewRepresentable {
    let paths: [(branch: String, path: [[Int]], controlPoints: [[Int]])]
    let spacingMM: [Double]  // (z, y, x)
    // world-mm position of voxel (0,0,0) — the server's own point_mm for
    // the wall-contour previews is origin_mm + voxel*spacingMM (see
    // server/path_series.py's voxel_path_to_world_mm), so the centerline/
    // control-point markers built from raw voxel indices need the same
    // origin added before converting to scene space, or they land offset
    // from the wall preview by a constant vector instead of right on it.
    let originMM: [Double]  // (z, y, x)
    // branch -> (path index -> raw contour, world-mm [z,y,x]) — same
    // shape segmentCurrentPoint accumulates in AppState.
    var wallContoursByBranch: [String: [Int: [[Double]]]] = [:]
    // The control point currently being dragged in the side panel's
    // reorder list, if any — highlighted with a white halo (not a color
    // change) so it's obvious which "knee" is about to move without
    // losing track of which branch it belongs to.
    var highlightBranch: String?
    var highlightIndex: Int?
    var controller: Scene3DController?

    func makeCoordinator() -> PathPreview3DCoordinator { PathPreview3DCoordinator() }

    func makeNSView(context: Context) -> Interactive3DSceneView {
        let view = Interactive3DSceneView()
        view.axisTag = "path3d"
        view.scene = SCNScene()
        view.backgroundColor = .black
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true

        let handler = context.coordinator.dragHandler
        view.onRightMouseDown = { [weak handler] point in handler?.rightMouseDown(point) }
        view.onRightMouseDragged = { [weak handler, weak view] point in
            guard let view else { return }
            handler?.rightMouseDragged(point, in: view)
        }
        view.onRightMouseUp = { [weak handler] in handler?.rightMouseUp() }
        view.onMiddleMouseDown = { [weak handler] point in handler?.middleMouseDown(point) }
        view.onMiddleMouseDragged = { [weak handler, weak view] point in
            guard let view else { return }
            handler?.middleMouseDragged(point, in: view)
        }
        view.onMiddleMouseUp = { [weak handler] in handler?.middleMouseUp() }
        return view
    }

    private func vector(_ voxel: [Double]) -> SCNVector3 {
        mmToScene([
            originMM[0] + voxel[0] * spacingMM[0],
            originMM[1] + voxel[1] * spacingMM[1],
            originMM[2] + voxel[2] * spacingMM[2],
        ])
    }

    /// Cheap fingerprint of everything that affects what gets drawn — see
    /// SegmentPreview3DView.signature() for why this matters: without it,
    /// every unrelated AppState publish (e.g. scrolling a slice slider)
    /// reconstructs this view and queues a full scene rebuild, and a
    /// backlog of queued no-op rebuilds was a real, confirmed cause of an
    /// unrelated slider feeling broken this session.
    private func signature() -> Int {
        var hasher = Hasher()
        for (branch, path, controlPoints) in paths {
            hasher.combine(branch)
            hasher.combine(path.count)
            if let first = path.first { hasher.combine(first) }
            if let last = path.last { hasher.combine(last) }
            hasher.combine(controlPoints)
        }
        for key in wallContoursByBranch.keys.sorted() {
            hasher.combine(key)
            for idx in (wallContoursByBranch[key]?.keys.sorted() ?? []) {
                hasher.combine(idx)
                hasher.combine(wallContoursByBranch[key]?[idx]?.count ?? 0)
            }
        }
        hasher.combine(highlightBranch)
        hasher.combine(highlightIndex)
        return hasher.finalize()
    }

    func updateNSView(_ view: Interactive3DSceneView, context: Context) {
        let sig = signature()
        guard context.coordinator.lastSignature != sig else { return }
        context.coordinator.lastSignature = sig
        // SwiftUI can call updateNSView from inside AppKit's own layout/
        // constraints pass (NSHostingView drives SwiftUI's render cycle
        // from its own layout override) — mutating an SCNView's scene
        // graph synchronously there can make it invalidate its own
        // layout in turn, which AppKit treats as illegal re-entrancy and
        // crashes with "_postWindowNeedsUpdateConstraints" (confirmed via
        // a real crash report, triggered by placing a centerline point
        // while this view was on screen). Deferring the actual rebuild to
        // the next run loop turn keeps it fully outside any layout pass.
        DispatchQueue.main.async { [self] in
            self.rebuildScene(view, context: context)
        }
    }

    private func rebuildScene(_ view: Interactive3DSceneView, context: Context) {
        guard let scene = view.scene else { return }
        let coordinator = context.coordinator
        // Rebuild path/marker geometry on every update, but never touch
        // the camera node — it's the same node across updates (unless this
        // is the very first build), so any zoom/pan/orbit the user already
        // applied survives an unrelated state change elsewhere in the app.
        scene.rootNode.childNodes.forEach {
            if $0 !== coordinator.cameraNode { $0.removeFromParentNode() }
        }

        let content = SCNNode()
        var fadeInNodes: [SCNNode] = []
        var hasAny = false
        for (branch, path, controlPoints) in paths {
            guard path.count > 1 else { continue }
            hasAny = true
            let color = NSColor(branchColor(branch))
            let smoothed = smoothPathPoints(path, controlPoints: controlPoints)
            let vectors = smoothed.map { vector($0) }
            content.addChildNode(lineNode(points: vectors, color: color))

            // "Knee" markers — the control points the user actually placed,
            // not the dense stitched path — so it's clear where they'd
            // drag/delete from in the 2D views too.
            for (cpIndex, cp) in controlPoints.enumerated() {
                let isHighlighted = branch == highlightBranch && cpIndex == highlightIndex
                let position = vector(cp.map(Double.init))

                if isHighlighted {
                    let halo = SCNSphere(radius: 1.4)
                    halo.firstMaterial?.diffuse.contents = NSColor.white
                    halo.firstMaterial?.lightingModel = .constant
                    halo.firstMaterial?.transparency = 0.4
                    halo.firstMaterial?.writesToDepthBuffer = false
                    let haloNode = SCNNode(geometry: halo)
                    haloNode.position = position
                    haloNode.renderingOrder = -1
                    content.addChildNode(haloNode)
                }

                let sphere = SCNSphere(radius: isHighlighted ? 0.85 : 0.6)
                sphere.firstMaterial?.diffuse.contents = color
                sphere.firstMaterial?.lightingModel = .constant
                let node = SCNNode(geometry: sphere)
                node.position = position
                content.addChildNode(node)
            }

            // The preliminary vessel-wall preview built during
            // segmentation (SegmentPathView) — stays visible here after
            // that sheet closes, fading in the first time it shows up so
            // it reads as "arriving", not an abrupt pop-in.
            let wallContours = wallContoursByBranch[branch] ?? [:]
            if !wallContours.isEmpty, let wallNode = buildWallMesh(contoursByIndex: wallContours, color: color) {
                wallNode.opacity = 0
                content.addChildNode(wallNode)
                fadeInNodes.append(wallNode)
            }
        }
        scene.rootNode.addChildNode(content)
        guard hasAny else { return }

        for node in fadeInNodes {
            // Animate the node's own opacity multiplier back up to fully
            // present (1.0) — the muted look itself already comes from
            // the wall mesh material's own transparency, set once in
            // buildWallMesh(); this just fades the node in on top of that.
            node.runAction(.fadeOpacity(to: 1.0, duration: 0.6))
        }

        let (center, radiusFloat) = content.boundingSphere
        let radius = CGFloat(radiusFloat)
        controller?.attach(view: view, center: center, radius: radius)

        // The far clip plane has to track the content's *current* size on
        // every rebuild, not just whatever it was when the camera was
        // first created — otherwise, as the path/wall grows past that
        // original radius, parts of it end up beyond zFar and silently
        // vanish (clipped, not actually missing) once the camera moves
        // far enough or at the wrong angle. This was a real, confirmed
        // bug: "areas disappear when you move the camera" after the path
        // grew past its first few points.
        coordinator.cameraNode?.camera?.zFar = Double(radius) * 10 + 100

        let totalControlPoints = paths.reduce(0) { $0 + $1.controlPoints.count }
        let totalWallContours = wallContoursByBranch.values.reduce(0) { $0 + $1.count }
        let shouldReframe = coordinator.cameraNode == nil
            || totalControlPoints != coordinator.lastControlPointCount
            || totalWallContours != coordinator.lastWallContourCount
        coordinator.lastControlPointCount = totalControlPoints
        coordinator.lastWallContourCount = totalWallContours
        guard shouldReframe else { return }

        let distance = radius * 2.5 + 10
        let cameraNode: SCNNode
        if let existing = coordinator.cameraNode {
            cameraNode = existing
        } else {
            cameraNode = SCNNode()
            let camera = SCNCamera()
            camera.zFar = Double(radius) * 10 + 100
            cameraNode.camera = camera
            coordinator.cameraNode = cameraNode
            scene.rootNode.addChildNode(cameraNode)
            view.pointOfView = cameraNode
        }
        cameraNode.position = SCNVector3(center.x, center.y, center.z + distance)
        cameraNode.look(at: center)
        view.defaultCameraController.target = center
    }
}
