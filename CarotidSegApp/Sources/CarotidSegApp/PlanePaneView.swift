import SwiftUI

/// Computes where an aspect-fit image actually lands inside a larger
/// container — needed to convert a tap location (in container coordinates)
/// to a normalized position on the image, and to place the crosshair
/// overlay at the right spot, since `.aspectRatio(contentMode: .fit)` can
/// letterbox the image within its frame.
/// Standard MPR color convention (matches 3D Slicer): each plane gets a
/// fixed color, used both for that plane's own label and for the crosshair
/// line drawn on the *other* two panes at that plane's current position —
/// e.g. the axial pane's horizontal crosshair line (showing the coronal
/// slice position) is colored with coronal's color, not axial's.
extension PlaneAxis {
    var color: Color {
        switch self {
        case .axial: return .red
        case .coronal: return .green
        case .sagittal: return .yellow
        }
    }
}

/// Centerline branch colors — deliberately distinct from the red/green/
/// yellow already used for the axis crosshairs, so a traced path never
/// reads as "part of the crosshair." Each branch is a full path from the
/// common carotid trunk to its own tip (cca_ica -> ICA, cca_eca -> ECA),
/// not just the distal segment, so the two paths deliberately overlap
/// along the shared trunk.
func branchColor(_ branch: String) -> Color {
    switch branch {
    // Same saturated red the old separate "cca" trunk segment used —
    // kept on cca_ica specifically (not cca_eca) since it's still the
    // path that starts at, and visually reads as, the common carotid.
    case "cca_ica": return Color(red: 0.95, green: 0.11, blue: 0.16)
    case "cca_eca": return .cyan
    default: return .white
    }
}

/// Human-readable label for a branch key, e.g. "cca_ica" -> "CCA → ICA".
func branchDisplayName(_ branch: String) -> String {
    switch branch {
    case "cca_ica": return "CCA → ICA"
    case "cca_eca": return "CCA → ECA"
    default: return branch.uppercased()
    }
}

/// `zoom` scales the fitted image about the container's center. Used both
/// to actually position/size the displayed image and to map clicks/overlay
/// coordinates — one shared calculation, so zooming can never desync the
/// crosshair, traced-path overlay, or click-to-voxel math from what's
/// actually on screen (they'd each silently drift apart if computed
/// independently).
/// Converts a slice image's raw pixel size into its true physical (mm)
/// aspect ratio for display — axial slices have isotropic in-plane
/// spacing so this is a no-op in practice there, but coronal/sagittal mix
/// the coarse through-plane spacing (slice thickness) with fine in-plane
/// spacing on their two axes, and rendering the raw pixel grid 1:1
/// visibly stretches/squashes the anatomy. Feeding this (not the raw
/// pixel size) into fitRect makes the Image's displayed frame stretch by
/// exactly the right non-uniform factor to correct it — and since every
/// overlay (crosshair, markers) and click/voxel mapping already goes
/// through that same fitRect, they all stay correct automatically.
func physicalImageSize(pixelSize: CGSize, axis: PlaneAxis, spacingMM: [Double]) -> CGSize {
    guard spacingMM.count == 3 else { return pixelSize }
    let rowSpacing: Double
    let colSpacing: Double
    switch axis {
    case .axial: (rowSpacing, colSpacing) = (spacingMM[1], spacingMM[2])      // rows=y, cols=x
    case .coronal: (rowSpacing, colSpacing) = (spacingMM[0], spacingMM[2])    // rows=z, cols=x
    case .sagittal: (rowSpacing, colSpacing) = (spacingMM[0], spacingMM[1])   // rows=z, cols=y
    }
    return CGSize(width: pixelSize.width * colSpacing, height: pixelSize.height * rowSpacing)
}

func fitRect(containerSize: CGSize, imageSize: CGSize, zoom: CGFloat = 1.0, pan: CGSize = .zero) -> CGRect {
    guard imageSize.width > 0, imageSize.height > 0, containerSize.width > 0, containerSize.height > 0 else {
        return CGRect(origin: .zero, size: containerSize)
    }
    let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height) * zoom
    let fitted = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    let origin = CGPoint(
        x: (containerSize.width - fitted.width) / 2 + pan.width,
        y: (containerSize.height - fitted.height) / 2 + pan.height
    )
    return CGRect(origin: origin, size: fitted)
}

/// One orthogonal plane view (axial/coronal/sagittal), with its own slice
/// slider — used three at once in SliceView's grid so all three projections
/// are visible simultaneously, the way SimVascular and other MPR viewers
/// lay them out. Clicking/dragging in any pane moves the shared 3D cursor
/// (AppState.axialIndex/coronalIndex/sagittalIndex), which scrolls the
/// other two panes to match and draws a crosshair here showing where they
/// currently are — standard MPR sync behavior.
struct PlanePaneView: View {
    let axis: PlaneAxis
    @EnvironmentObject var state: AppState
    @State private var image: NSImage?
    @State private var isLoading = false
    @State private var fetchError: String?
    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero

    /// Single source of truth for the slice-index Slider's valid upper
    /// bound — read by both its `in:` range and its clamped-get Binding
    /// (see the Slider below for why both must use the exact same
    /// expression). Floored at 1, not 0: a real crash (SIGTRAP in
    /// SwiftUI's Normalizing.init, which divides by (max-min) to
    /// normalize the slider's position) happened even with the value
    /// correctly clamped, which pointed to the *range itself* being
    /// degenerate (0...0) rather than a value/range mismatch — this
    /// pane's body got evaluated at least once while state.caseInfo was
    /// nil (sliceCount(for:) falls back to 1, so this was 1-1=0) despite
    /// SliceView gating PlanePaneView behind `if let info = state.caseInfo`,
    /// evidently not a hard guarantee against a stray AttributeGraph
    /// evaluation. A zero-width range crashes Slider outright regardless
    /// of the value; flooring at 1 makes that impossible.
    private var sliceCount: Int { max(state.sliceCount(for: axis) - 1, 1) }

    private func displaySize(for img: NSImage) -> CGSize {
        guard let spacing = state.caseInfo?.spacingMM else { return img.size }
        return physicalImageSize(pixelSize: img.size, axis: axis, spacingMM: spacing)
    }

    private let minZoom: CGFloat = 0.5
    private let maxZoom: CGFloat = 10.0
    private let depthToleranceVoxels = 2
    private let hitTestTolerancePt: CGFloat = 12

    var body: some View {
        VStack(spacing: 6) {
            Text(axis.label)
                .font(.caption)
                .foregroundStyle(axis.color)

            GeometryReader { geo in
                ZStack {
                    Color.black
                    if let image {
                        let rect = fitRect(containerSize: geo.size, imageSize: displaySize(for: image), zoom: zoomScale, pan: panOffset)
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                            // CONFIRMED via live UI automation this
                            // session (posting real mouse events and
                            // inspecting the accessibility tree): at high
                            // zoom, this Image's hit-testable/accessible
                            // bounds balloon far past the pane's visible
                            // box and silently swallow clicks meant for
                            // completely different sibling controls below
                            // (the slice-step buttons) — .clipped() only
                            // hides the visual overflow, it does nothing
                            // for hit-testing. Tried switching to
                            // scaleEffect (a render-only transform, in
                            // theory keeping the layout frame unzoomed) —
                            // re-tested the same way, and it *still*
                            // ballooned the same way, so the transform
                            // mechanism isn't what matters here. This
                            // Image has no gestures of its own — all real
                            // interaction already goes through the
                            // InteractivePlaneSurface overlay below — so
                            // just taking it out of hit-testing entirely
                            // is the actual fix, independent of size.
                            .allowsHitTesting(false)
                    }
                    if let image, let cross = state.crosshairNormalized(for: axis) {
                        crosshair(containerSize: geo.size, imageSize: displaySize(for: image), normalized: cross)
                            .allowsHitTesting(false)
                    }
                    if state.isMarkingCenterline, let image {
                        controlPointMarkers(containerSize: geo.size, imageSize: displaySize(for: image))
                            .allowsHitTesting(false)
                    }
                    if isLoading {
                        ProgressView()
                            .allowsHitTesting(false)
                    }
                }
                .clipped()
                .overlay(
                    GeometryReader { overlayGeo in
                    InteractivePlaneSurface(
                        axisTag: axis.rawValue,
                        globalFrame: overlayGeo.frame(in: .global),
                        onMouseDown: { location, size, clickCount in
                            handlePointerDown(at: location, containerSize: size, clickCount: clickCount)
                        },
                        onMouseDragged: { location, size in
                            handlePointerDragged(at: location, containerSize: size)
                        },
                        onMouseUp: { location, size in
                            handlePointerUp(at: location, containerSize: size)
                        },
                        onScrollStep: { step in
                            state.setIndex(state.index(for: axis) + step, for: axis)
                        },
                        onRightDrag: { deltaY in
                            // Drag up = zoom in, drag down = zoom out — the
                            // common DICOM-viewer right-drag convention.
                            zoomScale = min(max(zoomScale - deltaY * 0.01, minZoom), maxZoom)
                        },
                        onMiddleDrag: { delta in
                            panOffset.width += delta.width
                            panOffset.height += delta.height
                        }
                    )
                    }
                )
            }
            .frame(minWidth: 220, maxWidth: .infinity, minHeight: 220, maxHeight: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Native slider — safe now that the Image/overlay content
            // above no longer balloons its hit-testable frame at zoom
            // and intercepts clicks meant for this control.
            Slider(
                value: Binding(
                    get: { Double(min(max(state.index(for: axis), 0), sliceCount)) },
                    set: { state.setIndex(Int($0), for: axis) }
                ),
                in: 0...Double(sliceCount),
                step: 1
            )
            Text("\(state.index(for: axis)) / \(sliceCount)")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let fetchError {
                Text(fetchError)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Theme.panel))
        .task(id: fetchKey) {
            await fetchSlice()
        }
    }

    /// Which other axis each crosshair line on this pane represents — the
    /// vertical line's x position comes from `xAxis`'s current index, the
    /// horizontal line's y position from `yAxis`'s (same pairing
    /// updateCursorFromClick/crosshairNormalized use in AppState).
    private var otherAxes: (xAxis: PlaneAxis, yAxis: PlaneAxis) {
        switch axis {
        case .axial: return (.sagittal, .coronal)
        case .coronal: return (.sagittal, .axial)
        case .sagittal: return (.coronal, .axial)
        }
    }

    @ViewBuilder
    private func crosshair(containerSize: CGSize, imageSize: CGSize, normalized: (x: Double, y: Double)) -> some View {
        let rect = fitRect(containerSize: containerSize, imageSize: imageSize, zoom: zoomScale, pan: panOffset)
        let px = rect.origin.x + normalized.x * rect.width
        let py = rect.origin.y + normalized.y * rect.height
        let (xAxis, yAxis) = otherAxes

        Path { path in
            path.move(to: CGPoint(x: px, y: rect.minY))
            path.addLine(to: CGPoint(x: px, y: rect.maxY))
        }
        .stroke(xAxis.color.opacity(0.8), lineWidth: 0.75)

        Path { path in
            path.move(to: CGPoint(x: rect.minX, y: py))
            path.addLine(to: CGPoint(x: rect.maxX, y: py))
        }
        .stroke(yAxis.color.opacity(0.8), lineWidth: 0.75)
    }


    /// Markers for the selected branch's placed control points — only the
    /// ones close to the slice currently being viewed (not the whole
    /// path), so a point shows up right where you placed it without
    /// cluttering every other slice. While a point is being dragged, its
    /// marker follows the live (uncommitted) position instead.
    @ViewBuilder
    private func controlPointMarkers(containerSize: CGSize, imageSize: CGSize) -> some View {
        let rect = fitRect(containerSize: containerSize, imageSize: imageSize, zoom: zoomScale, pan: panOffset)
        let key = state.centerlineKey(side: state.selectedSide, branch: state.selectedBranch)
        let points = state.controlPoints[key] ?? []
        ForEach(Array(points.enumerated()), id: \.offset) { idx, voxel in
            let isDragging = state.draggingPointKey == key && state.draggingPointIndex == idx
            let displayVoxel = isDragging ? (state.draggingPointVoxel ?? voxel) : voxel
            if abs(displayVoxel[axis.shapeIndex] - state.index(for: axis)) <= depthToleranceVoxels,
               let proj = state.normalizedProjection(axis: axis, voxel: displayVoxel) {
                let markerPos = CGPoint(x: rect.origin.x + proj.x * rect.width, y: rect.origin.y + proj.y * rect.height)
                Circle()
                    .fill(branchColor(state.selectedBranch))
                    .overlay(Circle().stroke(Color.white, lineWidth: 1.5))
                    .frame(width: isDragging ? 14 : 10, height: isDragging ? 14 : 10)
                    .position(markerPos)
            }
        }
    }

    /// Index of the nearest *visible* (on this slice, within
    /// depthToleranceVoxels) control point marker to a click location, if
    /// any is within hitTestTolerancePt — used to decide whether a click
    /// should pick up/delete an existing point instead of adding a new one.
    private func nearestVisibleControlPoint(at location: CGPoint, rect: CGRect) -> Int? {
        let key = state.centerlineKey(side: state.selectedSide, branch: state.selectedBranch)
        let points = state.controlPoints[key] ?? []
        var best: (index: Int, dist: CGFloat)?
        for (idx, voxel) in points.enumerated() {
            guard abs(voxel[axis.shapeIndex] - state.index(for: axis)) <= depthToleranceVoxels,
                  let proj = state.normalizedProjection(axis: axis, voxel: voxel) else { continue }
            let px = rect.origin.x + proj.x * rect.width
            let py = rect.origin.y + proj.y * rect.height
            let d = hypot(location.x - px, location.y - py)
            if d <= hitTestTolerancePt, (best == nil || d < best!.dist) {
                best = (idx, d)
            }
        }
        return best?.index
    }

    private func handlePointerDown(at location: CGPoint, containerSize: CGSize, clickCount: Int) {
        guard let image else { return }
        let rect = fitRect(containerSize: containerSize, imageSize: displaySize(for: image), zoom: zoomScale, pan: panOffset)
        guard rect.width > 0, rect.height > 0 else { return }

        if state.isMarkingCenterline, let idx = nearestVisibleControlPoint(at: location, rect: rect) {
            if clickCount >= 2 {
                let side = state.selectedSide
                let branch = state.selectedBranch
                Task { await state.deleteCenterlinePoint(side: side, branch: branch, index: idx) }
            } else {
                let key = state.centerlineKey(side: state.selectedSide, branch: state.selectedBranch)
                state.draggingPointKey = key
                state.draggingPointIndex = idx
                state.draggingPointVoxel = state.controlPoints[key]?[idx]
            }
            return
        }
        performClick(at: location, rect: rect)
    }

    private func handlePointerDragged(at location: CGPoint, containerSize: CGSize) {
        guard let image else { return }
        let rect = fitRect(containerSize: containerSize, imageSize: displaySize(for: image), zoom: zoomScale, pan: panOffset)
        guard rect.width > 0, rect.height > 0 else { return }
        let nx = (location.x - rect.origin.x) / rect.width
        let ny = (location.y - rect.origin.y) / rect.height

        if state.draggingPointIndex != nil {
            guard let voxel = state.voxelFromClick(axis: axis, nx: nx, ny: ny) else { return }
            state.draggingPointVoxel = voxel
            return
        }
        // Only scrub the cursor continuously in navigate mode — in marking
        // or auto-trace mode, each drag event would otherwise add a dense
        // run of near-duplicate centerline points (or spuriously fire the
        // auto-trace's second click) instead of one point per click.
        guard !state.isMarkingCenterline, !state.isAutoTracingCenterline else { return }
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return }
        state.updateCursorFromClick(axis: axis, nx: nx, ny: ny)
    }

    private func handlePointerUp(at location: CGPoint, containerSize: CGSize) {
        guard let key = state.draggingPointKey, let idx = state.draggingPointIndex,
              let voxel = state.draggingPointVoxel else { return }
        state.draggingPointKey = nil
        state.draggingPointIndex = nil
        state.draggingPointVoxel = nil
        let parts = key.split(separator: ":")
        guard parts.count == 2 else { return }
        let side = String(parts[0])
        let branch = String(parts[1])
        Task { await state.moveCenterlinePoint(side: side, branch: branch, index: idx, voxel: voxel) }
    }

    private func performClick(at location: CGPoint, rect: CGRect) {
        let nx = (location.x - rect.origin.x) / rect.width
        let ny = (location.y - rect.origin.y) / rect.height
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return }

        if state.isAutoTracingCenterline {
            guard let voxel = state.voxelFromClick(axis: axis, nx: nx, ny: ny) else { return }
            state.setIndex(voxel[0], for: .axial)
            state.setIndex(voxel[1], for: .coronal)
            state.setIndex(voxel[2], for: .sagittal)
            let side = state.selectedSide
            let branch = state.selectedBranch
            Task { await state.handleAutoTraceClick(side: side, branch: branch, voxel: voxel) }
        } else if state.isMarkingCenterline {
            guard let voxel = state.voxelFromClick(axis: axis, nx: nx, ny: ny) else { return }
            state.setIndex(voxel[0], for: .axial)
            state.setIndex(voxel[1], for: .coronal)
            state.setIndex(voxel[2], for: .sagittal)
            let side = state.selectedSide
            let branch = state.selectedBranch
            Task { await state.addCenterlinePoint(side: side, branch: branch, voxel: voxel) }
        } else {
            state.updateCursorFromClick(axis: axis, nx: nx, ny: ny)
        }
    }

    private var fetchKey: String {
        "\(state.caseInfo?.caseId ?? "")|\(axis.rawValue)|\(state.index(for: axis))|\(Int(state.windowLevel))|\(Int(state.windowWidth))"
    }

    private func fetchSlice() async {
        guard let caseId = state.caseInfo?.caseId else { return }
        let key = fetchKey

        // Scrolling very often revisits a recently-seen slice (back-and-
        // forth, or the other two panes re-requesting one already fetched
        // this session) — skip the network+decode round trip entirely when
        // we already have it. Shared across all three panes' instances
        // (keyed by axis too, so they never collide).
        if let cached = sliceImageCache.object(forKey: key as NSString) {
            image = cached
            fetchError = nil
            return
        }

        // Debounce: wait a brief moment before actually fetching. While
        // actively dragging the slider or scrubbing, fetchKey (and so
        // this .task's id) keeps changing, which cancels this sleep
        // before it ever reaches the network — so a fast drag no longer
        // fires a request per pixel of movement (confirmed via logging:
        // a third or more of requests were getting cancelled mid-flight
        // during normal dragging), it fires one once the value actually
        // settles.
        try? await Task.sleep(nanoseconds: 70_000_000)
        guard !Task.isCancelled else { return }

        isLoading = true
        fetchError = nil
        defer { isLoading = false }
        do {
            let fetched = try await state.api.fetchSlice(
                caseId: caseId, axis: axis, index: state.index(for: axis),
                wl: state.windowLevel, ww: state.windowWidth
            )
            sliceImageCache.setObject(fetched, forKey: key as NSString)
            image = fetched
        } catch is CancellationError {
            // Superseded by a newer fetch (e.g. the user kept scrolling) —
            // expected during normal use, not a real failure worth showing.
        } catch let urlError as URLError where urlError.code == .cancelled {
            // Same thing, surfaced as a URLError instead when the
            // cancellation happens at the URLSession level.
        } catch {
            fetchError = "Failed: \(error.localizedDescription)"
        }
    }
}

/// Shared across every PlanePaneView instance (axial/coronal/sagittal all
/// draw from and populate the same cache; their keys never collide since
/// fetchKey includes the axis). Capped well above what three panes'
/// worth of normal back-and-forth scrolling needs, with plenty of margin,
/// while still bounding memory for a long session.
private let sliceImageCache: NSCache<NSString, NSImage> = {
    let cache = NSCache<NSString, NSImage>()
    cache.countLimit = 300
    return cache
}()
