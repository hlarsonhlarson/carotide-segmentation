import SceneKit
import SwiftUI

@MainActor
final class SegmentPreview3DCoordinator {
    var cameraNode: SCNNode?
    var hasFramed = false
    // Re-frame whenever the wall mesh's point count grows, not just on
    // the very first build — see PathPreview3DCoordinator.
    // lastWallContourCount for the identical bug this fixes here too:
    // without it, the camera stays at whatever tiny framing it had when
    // this view first appeared (often just a lone cross-section ring),
    // and the wall growing as more points get accepted never pulls the
    // camera back out to actually show it.
    var lastWallContourCount = -1
    var lastSignature: Int?
    let dragHandler = OrbitCameraDragHandler()
}

/// Lives next to the 2D oblique-slice segmentation editor (SegmentPathView):
/// shows the resampled Series centerline plus, growing as each point gets
/// accepted, a translucent preview of the vessel wall — a ribbon lofted
/// between every pair of consecutive accepted contours, in the branch's
/// color but muted (so it reads as "preview", not the final mesh).
struct SegmentPreview3DView: NSViewRepresentable {
    let branch: String
    let centerlineMM: [[Double]]       // dense Series points, world-mm [z,y,x]
    let contoursByIndex: [Int: [[Double]]]  // path index -> raw contour, world-mm [z,y,x]
    // Where the oblique slice currently being edited sits along the
    // centerline — drawn as a white ring so it's clear which level the 2D
    // editor is looking at right now. If that point already has an
    // accepted contour, the ring is traced along *that* (so it visibly
    // lies on the wall mesh instead of floating at a synthetic radius);
    // otherwise it falls back to a plain circle at currentRadiusMM.
    var currentIndex: Int?
    var currentPointMM: [Double]?
    var currentTangentMM: [Double]?
    var currentRadiusMM: Double = 2.5
    var controller: Scene3DController?

    private static let ringResolution = 18

    func makeCoordinator() -> SegmentPreview3DCoordinator { SegmentPreview3DCoordinator() }

    func makeNSView(context: Context) -> Interactive3DSceneView {
        let view = Interactive3DSceneView()
        view.axisTag = "segment-preview-3d"
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

    /// Cheap fingerprint of everything that affects what gets drawn —
    /// skips rebuilding the scene when nothing relevant actually changed.
    /// Without this, every unrelated AppState publish elsewhere in the app
    /// (e.g. scrolling a slice slider) reconstructs this struct and
    /// queues a full scene rebuild via updateNSView's deferred dispatch;
    /// with many queued no-op rebuilds piling up, the main thread can
    /// fall behind and make unrelated UI (that axial slider) feel broken
    /// — confirmed as a real regression this session, not a hypothetical.
    private func signature() -> Int {
        var hasher = Hasher()
        hasher.combine(branch)
        hasher.combine(centerlineMM.count)
        for idx in contoursByIndex.keys.sorted() {
            hasher.combine(idx)
            hasher.combine(contoursByIndex[idx]?.count ?? 0)
        }
        hasher.combine(currentIndex)
        hasher.combine(currentPointMM ?? [])
        hasher.combine(currentTangentMM ?? [])
        hasher.combine(currentRadiusMM)
        return hasher.finalize()
    }

    func updateNSView(_ view: Interactive3DSceneView, context: Context) {
        let sig = signature()
        guard context.coordinator.lastSignature != sig else { return }
        context.coordinator.lastSignature = sig
        // See PathPreview3DView.updateNSView: SwiftUI can call this from
        // inside AppKit's own layout pass, and mutating an SCNView's scene
        // graph synchronously there risks a real, confirmed crash
        // (_postWindowNeedsUpdateConstraints re-entrancy). Defer to the
        // next run loop turn to stay outside any layout pass.
        DispatchQueue.main.async { [self] in
            self.rebuildScene(view, context: context)
        }
    }

    private func rebuildScene(_ view: Interactive3DSceneView, context: Context) {
        guard let scene = view.scene else { return }
        let coordinator = context.coordinator
        scene.rootNode.childNodes.forEach {
            if $0 !== coordinator.cameraNode { $0.removeFromParentNode() }
        }

        let content = SCNNode()
        var hasAny = false

        if centerlineMM.count > 1 {
            hasAny = true
            let color = NSColor(branchColor(branch))
            content.addChildNode(lineNode(points: centerlineMM.map(mmToScene), color: color))
        }

        // A gap (an un-segmented point in between two accepted ones) just
        // means no ribbon spans that gap — the correct "still growing"
        // look rather than bridging over a point that hasn't been
        // reviewed yet (buildWallMesh already only connects strictly
        // consecutive indices).
        let wallColor = NSColor(branchColor(branch))
        if let wallNode = buildWallMesh(contoursByIndex: contoursByIndex, color: wallColor, ringResolution: Self.ringResolution) {
            hasAny = true
            content.addChildNode(wallNode)
        }

        // Prefer tracing the ring along this point's own accepted contour
        // (so it visibly sits on the wall mesh) over a synthetic circle,
        // which is only a fallback for a point that hasn't been segmented
        // (or seeded) yet.
        if let currentIndex, let raw = contoursByIndex[currentIndex] {
            let ring = resampleClosedLoopMM(raw, count: 40)
            if !ring.isEmpty {
                hasAny = true
                let node = closedLoopNode(points: ring, color: .white)
                node.renderingOrder = 10
                content.addChildNode(node)
            }
        } else if let currentPointMM, let currentTangentMM {
            hasAny = true
            content.addChildNode(crossSectionRingNode(
                centerMM: currentPointMM, tangentMM: currentTangentMM, radiusMM: currentRadiusMM, color: .white
            ))
        }

        scene.rootNode.addChildNode(content)
        guard hasAny else { return }

        let (center, radiusFloat) = content.boundingSphere
        let radius = CGFloat(radiusFloat)
        controller?.attach(view: view, center: center, radius: radius)

        // Keep the far clip plane current even on rebuilds that don't
        // reframe — see PathPreview3DView's identical fix: a stale zFar
        // from the first (smaller) build makes later, larger content
        // silently vanish past it once the camera moves.
        coordinator.cameraNode?.camera?.zFar = Double(radius) * 10 + 100

        let shouldReframe = !coordinator.hasFramed || contoursByIndex.count != coordinator.lastWallContourCount
        coordinator.lastWallContourCount = contoursByIndex.count
        guard shouldReframe else { return }
        coordinator.hasFramed = true

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
    }
}
