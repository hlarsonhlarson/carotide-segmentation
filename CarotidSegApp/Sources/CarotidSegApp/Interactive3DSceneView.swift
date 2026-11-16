import SceneKit
import AppKit

/// SCNView subclass shared by every 3D preview pane in the app. Left-drag-
/// orbit and scroll-zoom come for free from SceneKit's own
/// `allowsCameraControl`; this class only adds the window-level plumbing
/// (via PaneEventRouter) needed for right-drag-zoom and middle-drag-pan,
/// matching the convention used in the 2D panes. `axisTag` must be unique
/// per on-screen instance (PaneEventRouter keys registrations by it).
final class Interactive3DSceneView: SCNView {
    var axisTag: String = "path3d"
    var onRightMouseDown: ((CGPoint) -> Void)?
    var onRightMouseDragged: ((CGPoint) -> Void)?
    var onRightMouseUp: (() -> Void)?
    var onMiddleMouseDown: ((CGPoint) -> Void)?
    var onMiddleMouseDragged: ((CGPoint) -> Void)?
    var onMiddleMouseUp: (() -> Void)?

    override var isFlipped: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // A cached scene can be populated before its sheet is attached.
        // Draw it on attachment without enabling continuous rendering.
        if window != nil { needsDisplay = true }
    }

    override func layout() {
        super.layout()
        guard let window else { return }
        let windowRect = convert(bounds, to: nil)
        PaneEventRouter.shared.register(
            tag: axisTag, window: window, windowFrame: windowRect,
            callbacks: .init(
                onMouseDown: { _, _, _ in },
                onMouseDragged: { _, _ in },
                onMouseUp: { _, _ in },
                onScroll: { _ in },
                onRightMouseDown: { [weak self] p in self?.onRightMouseDown?(p) },
                onRightMouseDragged: { [weak self] p in self?.onRightMouseDragged?(p) },
                onRightMouseUp: { [weak self] in self?.onRightMouseUp?() },
                onMiddleMouseDown: { [weak self] p in self?.onMiddleMouseDown?(p) },
                onMiddleMouseDragged: { [weak self] p in self?.onMiddleMouseDragged?(p) },
                onMiddleMouseUp: { [weak self] in self?.onMiddleMouseUp?() }
            ),
            // Leave SceneKit's own default handling alone for these —
            // left-drag-orbit and scroll-zoom already work fine.
            interceptsLeft: false,
            interceptsRight: true,
            interceptsMiddle: true,
            interceptsScroll: false
        )
    }

    override func viewWillMove(toWindow newWindow: NSWindow?) {
        super.viewWillMove(toWindow: newWindow)
        if newWindow == nil {
            PaneEventRouter.shared.unregister(tag: axisTag)
        }
    }
}

/// Drives right-drag-zoom and middle-drag-pan through the SCNView's own
/// `defaultCameraController` (the object `allowsCameraControl` itself
/// uses), rather than mutating the camera node directly.
///
/// Mutating the node directly was tried first and silently did nothing:
/// with `allowsCameraControl = true`, SceneKit hands the pointOfView node
/// to its own internal SCNCameraController, which keeps reasserting *its
/// own* internal distance/orientation state every frame (that's what
/// makes left-drag-orbit and scroll-zoom work for free) — poking the
/// node's transform from outside that controller gets overwritten on the
/// next frame. Driving the same controller's own `dolly`/
/// `translateInCameraSpaceBy` API instead keeps our input and its internal
/// state in sync.
@MainActor
final class OrbitCameraDragHandler {
    private var lastRight: CGPoint?
    private var lastMiddle: CGPoint?

    func rightMouseDown(_ point: CGPoint) {
        lastRight = point
    }

    func rightMouseDragged(_ point: CGPoint, in view: SCNView) {
        guard let last = lastRight else { return }
        // Drag up = zoom in, matching the 2D panes' right-drag convention
        // — isFlipped means screen-up is a *decrease* in y.
        let delta = Float((last.y - point.y) * 0.5)
        let center = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        view.defaultCameraController.dolly(by: delta, onScreenPoint: center, viewport: view.bounds.size)
        lastRight = point
    }

    func rightMouseUp() {
        lastRight = nil
    }

    func middleMouseDown(_ point: CGPoint) {
        lastMiddle = point
    }

    func middleMouseDragged(_ point: CGPoint, in view: SCNView) {
        guard let last = lastMiddle, let pov = view.pointOfView else { return }
        let dx = Float((point.x - last.x) * 0.15)
        let dy = Float((point.y - last.y) * 0.15)
        let controller = view.defaultCameraController

        // translateInCameraSpaceBy only moves the camera, not the
        // controller's orbit target — left unsynced, the target stays
        // behind at its pre-pan position, so left-drag-orbit afterward
        // rotates around a point that's no longer where the camera is
        // looking. Shift the target by the same world-space offset so
        // orbiting after a pan still feels anchored to what's on screen.
        let rightX = Float(pov.worldRight.x), rightY = Float(pov.worldRight.y), rightZ = Float(pov.worldRight.z)
        let upX = Float(pov.worldUp.x), upY = Float(pov.worldUp.y), upZ = Float(pov.worldUp.z)
        let worldDX: Float = -dx
        let worldDY: Float = dy
        let offsetX: Float = rightX * worldDX + upX * worldDY
        let offsetY: Float = rightY * worldDX + upY * worldDY
        let offsetZ: Float = rightZ * worldDX + upZ * worldDY
        let oldTarget = controller.target
        controller.target = SCNVector3(
            oldTarget.x + CGFloat(offsetX), oldTarget.y + CGFloat(offsetY), oldTarget.z + CGFloat(offsetZ)
        )
        controller.translateInCameraSpaceBy(x: -dx, y: dy, z: 0)
        lastMiddle = point
    }

    func middleMouseUp() {
        lastMiddle = nil
    }
}
