import AppKit

/// Routes every pointer interaction for the plane panes (scroll, click/
/// drag, right-drag-zoom, middle-drag-pan) through one application-wide
/// local event monitor, hit-testing a manually-tracked registry of
/// (window-space rect -> callbacks) pairs — instead of relying on each
/// pane's own NSView's AppKit-dispatched overrides.
///
/// VERIFIED this class of bug is real, not a guess: per-view scrollWheel
/// overrides worked for every pane right after a case loaded, but events
/// for the axial pane specifically stopped arriving at its view after a
/// short time in the session — while mouseDown/hitTest kept resolving to
/// that exact same view correctly for the rest of the session (confirmed
/// via hitTest logging in an earlier debugging pass). Later reports
/// showed the same flakiness spreading to clicks (point placement working
/// "every other time", unable to place a first point) — consistent with
/// the same underlying AppKit dispatch confusion, just surfacing on a
/// different event type. None of this is fixable from inside the
/// affected view's own event handlers, since the dispatch itself is what's
/// unreliable; hit-testing every event ourselves at the window level,
/// against a rect we update on every layout pass, sidesteps whatever
/// internal routing state AppKit is getting confused about.
///
/// A region can selectively opt out of interception per button (e.g. the
/// 3D preview wants SceneKit's own default left-drag-orbit and
/// scroll-zoom left alone, while we take over right/middle-drag
/// ourselves) — when a button isn't intercepted for a region, a
/// down/dragged/up sequence over it is left completely untouched so the
/// view's native handling (SceneKit's camera control, a native NSControl's
/// own tracking loop) sees it exactly as if we weren't here.
@MainActor
final class PaneEventRouter {
    static let shared = PaneEventRouter()

    struct Callbacks {
        var onMouseDown: (CGPoint, CGSize, Int) -> Void
        var onMouseDragged: (CGPoint, CGSize) -> Void
        var onMouseUp: (CGPoint, CGSize) -> Void
        var onScroll: (CGFloat) -> Void
        var onRightMouseDown: (CGPoint) -> Void
        var onRightMouseDragged: (CGPoint) -> Void
        var onRightMouseUp: () -> Void
        var onMiddleMouseDown: (CGPoint) -> Void
        var onMiddleMouseDragged: (CGPoint) -> Void
        var onMiddleMouseUp: () -> Void
    }

    private struct Region {
        var windowID: ObjectIdentifier
        var frame: CGRect
        var callbacks: Callbacks
        var interceptsLeft: Bool
        var interceptsRight: Bool
        var interceptsMiddle: Bool
        var interceptsScroll: Bool
    }

    private var regions: [String: Region] = [:]
    private var monitor: Any?

    // Which pane "owns" an in-progress drag, per button — so dragged/up
    // events keep routing to the same pane even if the cursor strays
    // outside its rect mid-drag, matching normal AppKit click-tracking
    // semantics (e.g. dragging a control point slightly past the edge).
    // Reset on every new *MouseDown event unconditionally (matched pane or
    // not) — leaving a stale tag set from a previous gesture was a real
    // bug: if a mouseUp was ever missed, every later drag anywhere in the
    // window (e.g. an unrelated Slider) would get silently hijacked and
    // consumed by whichever pane's tag was stuck active.
    private var activeLeftTag: String?
    private var activeRightTag: String?
    private var activeMiddleTag: String?

    private init() {
        monitor = NSEvent.addLocalMonitorForEvents(matching: [
            .scrollWheel,
            .leftMouseDown, .leftMouseDragged, .leftMouseUp,
            .rightMouseDown, .rightMouseDragged, .rightMouseUp,
            .otherMouseDown, .otherMouseDragged, .otherMouseUp,
        ]) { [weak self] event in
            self?.handle(event) ?? event
        }
    }

    func register(
        tag: String, window: NSWindow, windowFrame: CGRect, callbacks: Callbacks,
        interceptsLeft: Bool = true, interceptsRight: Bool = true,
        interceptsMiddle: Bool = true, interceptsScroll: Bool = true
    ) {
        regions[tag] = Region(
            windowID: ObjectIdentifier(window), frame: windowFrame, callbacks: callbacks,
            interceptsLeft: interceptsLeft, interceptsRight: interceptsRight,
            interceptsMiddle: interceptsMiddle, interceptsScroll: interceptsScroll
        )
    }

    func unregister(tag: String) {
        regions[tag] = nil
        if activeLeftTag == tag { activeLeftTag = nil }
        if activeRightTag == tag { activeRightTag = nil }
        if activeMiddleTag == tag { activeMiddleTag = nil }
    }

    /// Window coordinates are bottom-left-origin (y up); our panes report
    /// themselves as flipped (top-left origin, y down) to match SwiftUI's
    /// own convention, and every call site (fitRect, hit-testing) assumes
    /// that — so flip Y relative to the region's own frame here, once,
    /// rather than at every call site.
    private func localPoint(_ windowLocation: CGPoint, in frame: CGRect) -> CGPoint {
        CGPoint(x: windowLocation.x - frame.minX, y: frame.height - (windowLocation.y - frame.minY))
    }

    private func tag(at location: CGPoint, in windowID: ObjectIdentifier) -> String? {
        for (tag, region) in regions where region.windowID == windowID && region.frame.contains(location) {
            return tag
        }
        return nil
    }

    private func handle(_ event: NSEvent) -> NSEvent? {
        guard let window = event.window else { return event }
        let windowID = ObjectIdentifier(window)
        let location = event.locationInWindow

        switch event.type {
        case .scrollWheel:
            guard let tag = tag(at: location, in: windowID), let region = regions[tag], region.interceptsScroll else { return event }
            region.callbacks.onScroll(event.scrollingDeltaY)
            return nil

        case .leftMouseDown:
            let matched = tag(at: location, in: windowID)
            activeLeftTag = matched.flatMap { regions[$0]?.interceptsLeft == true ? $0 : nil }
            guard let tag = activeLeftTag, let region = regions[tag] else { return event }
            region.callbacks.onMouseDown(localPoint(location, in: region.frame), region.frame.size, event.clickCount)
            return nil

        case .leftMouseDragged:
            guard let tag = activeLeftTag, let region = regions[tag] else { return event }
            region.callbacks.onMouseDragged(localPoint(location, in: region.frame), region.frame.size)
            return nil

        case .leftMouseUp:
            guard let tag = activeLeftTag, let region = regions[tag] else { return event }
            activeLeftTag = nil
            region.callbacks.onMouseUp(localPoint(location, in: region.frame), region.frame.size)
            return nil

        case .rightMouseDown:
            let matched = tag(at: location, in: windowID)
            activeRightTag = matched.flatMap { regions[$0]?.interceptsRight == true ? $0 : nil }
            guard let tag = activeRightTag, let region = regions[tag] else { return event }
            region.callbacks.onRightMouseDown(localPoint(location, in: region.frame))
            return nil

        case .rightMouseDragged:
            guard let tag = activeRightTag, let region = regions[tag] else { return event }
            region.callbacks.onRightMouseDragged(localPoint(location, in: region.frame))
            return nil

        case .rightMouseUp:
            guard let tag = activeRightTag, let region = regions[tag] else { return event }
            activeRightTag = nil
            region.callbacks.onRightMouseUp()
            return nil

        case .otherMouseDown:
            guard event.buttonNumber == 2 else { return event }
            let matched = tag(at: location, in: windowID)
            activeMiddleTag = matched.flatMap { regions[$0]?.interceptsMiddle == true ? $0 : nil }
            guard let tag = activeMiddleTag, let region = regions[tag] else { return event }
            region.callbacks.onMiddleMouseDown(localPoint(location, in: region.frame))
            return nil

        case .otherMouseDragged:
            guard event.buttonNumber == 2, let tag = activeMiddleTag, let region = regions[tag] else { return event }
            region.callbacks.onMiddleMouseDragged(localPoint(location, in: region.frame))
            return nil

        case .otherMouseUp:
            guard event.buttonNumber == 2, let tag = activeMiddleTag, let region = regions[tag] else { return event }
            activeMiddleTag = nil
            region.callbacks.onMiddleMouseUp()
            return nil

        default:
            return event
        }
    }
}
