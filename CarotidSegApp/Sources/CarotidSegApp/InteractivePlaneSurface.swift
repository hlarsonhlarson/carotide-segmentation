import AppKit
import SwiftUI

/// Captures click/drag (for click-to-navigate the shared 3D cursor),
/// scroll-wheel (for scrolling through this pane's own slices), right-
/// drag-zoom and middle-drag-pan. All actual event *dispatch* goes through
/// PaneEventRouter (one window-wide monitor that hit-tests a tracked rect
/// per pane), not AppKit's own per-view event delivery to this NSView —
/// see PaneEventRouter's doc comment for why: AppKit's native dispatch to
/// this view was observed to become unreliable mid-session (scroll, then
/// later clicks too), while this view's own geometry stayed perfectly
/// normal throughout. This view still exists mainly to own per-pane drag
/// state (scroll accumulator, last right/middle-drag location) and to
/// keep re-registering its current on-screen rect with the router as it
/// resizes.
struct InteractivePlaneSurface: NSViewRepresentable {
    var axisTag: String = ""
    // This view's on-screen frame, measured by SwiftUI itself (pass
    // `proxy.frame(in: .global)` from a GeometryReader wrapping this view)
    // rather than by asking AppKit to `convert(bounds, to: nil)` on the
    // hosted NSView. The two were found to disagree by a consistent ~20pt
    // vertically on a real build (confirmed via a synthetic click placed
    // at a known screen pixel, then measuring the resulting marker's
    // on-screen position against it) even though their *sizes* matched —
    // i.e. a pure origin desync between AppKit's self-reported bounds and
    // where SwiftUI actually painted this view. Every other piece of this
    // pane's geometry (the image, crosshair, markers) is positioned via
    // SwiftUI's own GeometryReader, so feeding click hit-testing from that
    // same source instead of AppKit's own measurement is what keeps clicks
    // and rendering in agreement.
    var globalFrame: CGRect = .zero
    var onMouseDown: (CGPoint, CGSize, Int) -> Void  // location, container size, clickCount
    var onMouseDragged: (CGPoint, CGSize) -> Void
    var onMouseUp: (CGPoint, CGSize) -> Void
    var onScrollStep: (Int) -> Void
    var onRightDrag: (CGFloat) -> Void
    var onMiddleDrag: (CGSize) -> Void

    func makeNSView(context: Context) -> SurfaceView {
        let view = SurfaceView()
        view.axisTag = axisTag
        view.globalFrame = globalFrame
        view.onMouseDown = onMouseDown
        view.onMouseDragged = onMouseDragged
        view.onMouseUp = onMouseUp
        view.onScrollStep = onScrollStep
        view.onRightDrag = onRightDrag
        view.onMiddleDrag = onMiddleDrag
        return view
    }

    func updateNSView(_ nsView: SurfaceView, context: Context) {
        nsView.axisTag = axisTag
        nsView.globalFrame = globalFrame
        nsView.onMouseDown = onMouseDown
        nsView.onMouseDragged = onMouseDragged
        nsView.onMouseUp = onMouseUp
        nsView.onScrollStep = onScrollStep
        nsView.onRightDrag = onRightDrag
        nsView.onMiddleDrag = onMiddleDrag
        // SwiftUI calls this whenever `globalFrame` (or anything else)
        // changes — which is exactly when the registered hit-test rect
        // needs updating. layout() alone isn't a reliable enough signal:
        // a pure position change (no size change) doesn't always trigger
        // an AppKit layout pass.
        nsView.registerWithRouter()
    }

    final class SurfaceView: NSView {
        var axisTag: String = ""
        var globalFrame: CGRect = .zero
        var onMouseDown: ((CGPoint, CGSize, Int) -> Void)?
        var onMouseDragged: ((CGPoint, CGSize) -> Void)?
        var onMouseUp: ((CGPoint, CGSize) -> Void)?
        var onScrollStep: ((Int) -> Void)?
        var onRightDrag: ((CGFloat) -> Void)?
        var onMiddleDrag: ((CGSize) -> Void)?
        private var scrollAccumulator: CGFloat = 0
        private let scrollStepThreshold: CGFloat = 4.0
        private var lastRightDragLocation: CGPoint?
        private var lastMiddleDragLocation: CGPoint?

        // Top-left origin, matching SwiftUI's own coordinate convention
        // (the fitRect/tap-mapping math in PlanePaneView already assumes
        // this), instead of AppKit's default bottom-left.
        override var isFlipped: Bool { true }

        override func layout() {
            super.layout()
            // Re-register on every layout pass so the tracked rect stays
            // current as panes resize. See PaneEventRouter for why actual
            // dispatch goes through it instead of the overrides below.
            registerWithRouter()
        }

        func registerWithRouter() {
            guard let window else { return }
            // `globalFrame` is SwiftUI's .global space: top-left origin,
            // relative to the window's content view. Window-base
            // coordinates (what event.locationInWindow is expressed in,
            // and what PaneEventRouter's hit-testing assumes) are
            // bottom-left origin — flip using the content view's height,
            // not `window.frame.height`, since that includes the title
            // bar while the content view (and SwiftUI's .global space)
            // does not.
            let contentHeight = window.contentView?.bounds.height ?? window.frame.height
            let windowRect = CGRect(
                x: globalFrame.minX,
                y: contentHeight - globalFrame.maxY,
                width: globalFrame.width,
                height: globalFrame.height
            )
            PaneEventRouter.shared.register(
                tag: axisTag, window: window, windowFrame: windowRect,
                callbacks: .init(
                    onMouseDown: { [weak self] point, size, clickCount in
                        self?.onMouseDown?(point, size, clickCount)
                    },
                    onMouseDragged: { [weak self] point, size in
                        self?.onMouseDragged?(point, size)
                    },
                    onMouseUp: { [weak self] point, size in
                        self?.onMouseUp?(point, size)
                    },
                    onScroll: { [weak self] deltaY in
                        self?.handleScrollDelta(deltaY)
                    },
                    onRightMouseDown: { [weak self] point in
                        self?.lastRightDragLocation = point
                    },
                    onRightMouseDragged: { [weak self] point in
                        guard let self else { return }
                        if let last = self.lastRightDragLocation {
                            self.onRightDrag?(point.y - last.y)
                        }
                        self.lastRightDragLocation = point
                    },
                    onRightMouseUp: { [weak self] in
                        self?.lastRightDragLocation = nil
                    },
                    onMiddleMouseDown: { [weak self] point in
                        self?.lastMiddleDragLocation = point
                    },
                    onMiddleMouseDragged: { [weak self] point in
                        guard let self else { return }
                        if let last = self.lastMiddleDragLocation {
                            self.onMiddleDrag?(CGSize(width: point.x - last.x, height: point.y - last.y))
                        }
                        self.lastMiddleDragLocation = point
                    },
                    onMiddleMouseUp: { [weak self] in
                        self?.lastMiddleDragLocation = nil
                    }
                )
            )
        }

        override func viewWillMove(toWindow newWindow: NSWindow?) {
            super.viewWillMove(toWindow: newWindow)
            if newWindow == nil {
                PaneEventRouter.shared.unregister(tag: axisTag)
            }
        }

        private func handleScrollDelta(_ deltaY: CGFloat) {
            scrollAccumulator += deltaY
            while scrollAccumulator >= scrollStepThreshold {
                onScrollStep?(1)
                scrollAccumulator -= scrollStepThreshold
            }
            while scrollAccumulator <= -scrollStepThreshold {
                onScrollStep?(-1)
                scrollAccumulator += scrollStepThreshold
            }
        }
    }
}
