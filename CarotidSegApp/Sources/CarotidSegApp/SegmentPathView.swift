import SwiftUI

/// Step-through 2D segmentation along a marked path — the SimVascular
/// workflow this whole app exists to replicate: move to a path point, view
/// the oblique slice perpendicular to the vessel there, run Threshold
/// (region-growing from the seed against the real HU cliff — the default
/// for CT) or Level Set against the real image, accept (or retry, or
/// trace a ring by hand), advance. Every point starts pre-seeded by
/// AppState.seedCircularContours (run automatically right after the
/// Series is prepared) — a real detection attempt at each point, falling
/// back to a plain circle only where that doesn't look like a
/// trustworthy vessel cross-section — so there's always a complete wall
/// preview, even before anything is reviewed by hand; points where it's
/// still wrong can be fixed individually.
struct SegmentPathView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    let side: String
    let branch: String

    @State private var image: NSImage?
    @State private var isLoadingSlice = false
    @State private var showPreview = false
    @State private var zoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    // Vertices of the ring being traced by hand, in the oblique slice's
    // own pixel space (same [0, RESLICE_SIZE_PX] convention the server's
    // own contours use) — only used while segmentMethod == "manual".
    @State private var manualPoints: [CGPoint] = []
    // A separate WL/WW just for this view — CT's default window (tuned
    // for general navigation) often doesn't show the lumen/wall boundary
    // as clearly as a narrower, contrast-centered one does.
    @State private var segWindowLevel: Double = 150
    @State private var segWindowWidth: Double = 350
    @StateObject private var previewController = Scene3DController()

    private let minZoom: CGFloat = 0.5
    private let maxZoom: CGFloat = 10.0

    private var seriesKey: String { state.centerlineKey(side: side, branch: branch) }

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            editorPane
                .frame(minWidth: 640, minHeight: 700)

            if showPreview {
                previewPane
                    .frame(minWidth: 380, minHeight: 700)
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
        }
        .padding()
        .frame(minHeight: 700)
        // Done lives outside both panes entirely, in its own corner —
        // never sharing layout with (or being able to overlap) either
        // pane's own click-handling surface.
        .overlay(alignment: .topTrailing) {
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .padding(14)
        }
        .task(id: "\(state.currentPathIndex)|\(segWindowLevel)|\(segWindowWidth)") {
            await loadSlice()
        }
        .onAppear {
            if state.caseInfo?.modality != "CT" {
                segWindowLevel = state.windowLevel
                segWindowWidth = state.windowWidth
            }
            withAnimation(.easeInOut(duration: 0.4)) {
                showPreview = true
            }
        }
    }

    private var editorPane: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Segment \(side.capitalized) \(branchDisplayName(branch))")
                    .font(.headline)
                Spacer()
                if state.isSeedingCircles {
                    ProgressView().controlSize(.small)
                    Text("Seeding round contours…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Text("Point \(state.currentPathIndex + 1) / \(state.seriesPointCount)"
                 + (state.acceptedIndices.contains(state.currentPathIndex) ? "  ✓ segmented" : ""))
                .font(.caption)
                .foregroundStyle(.secondary)

            GeometryReader { geo in
                ZStack {
                    Color.black
                    if let image {
                        let rect = fitRect(containerSize: geo.size, imageSize: image.size, zoom: zoomScale, pan: panOffset)
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                    }
                    if !state.currentContourPx.isEmpty, let image {
                        contourOverlay(containerSize: geo.size, imageSize: image.size)
                    }
                    if !manualPoints.isEmpty, let image {
                        manualRingOverlay(containerSize: geo.size, imageSize: image.size)
                    }
                    if isLoadingSlice {
                        ProgressView()
                    }
                }
                .clipped()
                .overlay(
                    GeometryReader { overlayGeo in
                    InteractivePlaneSurface(
                        axisTag: "segment-slice",
                        globalFrame: overlayGeo.frame(in: .global),
                        onMouseDown: { location, size, _ in
                            guard state.segmentMethod == "manual", let image else { return }
                            if let p = imagePixelPoint(from: location, containerSize: size, imageSize: image.size) {
                                manualPoints.append(p)
                            }
                        },
                        onMouseDragged: { location, size in
                            // Freehand: keep appending points while the
                            // button stays down, not just one per click —
                            // but only once the cursor has moved far
                            // enough from the last point to avoid an
                            // overly dense ring from a slow drag.
                            guard state.segmentMethod == "manual", let image else { return }
                            guard let p = imagePixelPoint(from: location, containerSize: size, imageSize: image.size) else { return }
                            if let last = manualPoints.last, hypot(p.x - last.x, p.y - last.y) < 3 { return }
                            manualPoints.append(p)
                        },
                        onMouseUp: { _, _ in },
                        onScrollStep: { _ in },
                        onRightDrag: { deltaY in
                            // Drag up = zoom in, matching every other
                            // pane's right-drag convention.
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
            .frame(minWidth: 420, minHeight: 420)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                Button("◀ Prev") { Task { await step(-1) } }
                    .disabled(state.currentPathIndex == 0)
                Spacer()
                Button("Next ▶") { Task { await step(1) } }
                    .disabled(state.currentPathIndex >= state.seriesPointCount - 1)
            }

            if state.caseInfo?.modality == "CT" {
                HStack(spacing: 12) {
                    Text("Contrast").font(.caption).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Level \(Int(segWindowLevel))").font(.caption2)
                        Slider(value: $segWindowLevel, in: state.windowLevelRange)
                    }
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Width \(Int(segWindowWidth))").font(.caption2)
                        Slider(value: $segWindowWidth, in: state.windowWidthRange)
                    }
                    Button("Reset") { segWindowLevel = 150; segWindowWidth = 350 }
                        .font(.caption)
                }
            }

            Divider()

            Picker("Method", selection: $state.segmentMethod) {
                Text("Threshold").tag("threshold")
                Text("Level Set").tag("levelset")
                Text("Manual").tag("manual")
            }
            .pickerStyle(.segmented)
            .frame(width: 260)
            .onChange(of: state.segmentMethod) { _ in manualPoints = [] }

            if state.segmentMethod == "manual" {
                HStack(spacing: 12) {
                    Button("Undo Point") { manualPoints.removeLast() }
                        .disabled(manualPoints.isEmpty)
                    Spacer()
                    Button("Use This Ring") {
                        let contourPx = manualPoints.map { [Double($0.y), Double($0.x)] }
                        Task {
                            await state.submitManualRing(side: side, branch: branch, contourPx: contourPx)
                            manualPoints = []
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(manualPoints.count < 3)
                }
            } else {
                HStack(spacing: 12) {
                    VStack(alignment: .leading) {
                        Text("Radius: \(String(format: "%.1f", state.segmentRadiusMM)) mm")
                        Slider(value: $state.segmentRadiusMM, in: 0.5...10.0)
                    }
                    .frame(width: 220)

                    Button("Segment This Point") {
                        Task {
                            await state.segmentCurrentPoint(side: side, branch: branch)
                        }
                    }
                    .buttonStyle(.borderedProminent)

                    if let fitted = state.lastFittedRadiusMM {
                        Text("fitted: \(String(format: "%.2f", fitted)) mm")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()
                }
            }

            Button("Delete Segmentation", role: .destructive) {
                Task { await state.deleteCurrentSegmentation(side: side, branch: branch) }
            }
            .disabled(!state.acceptedIndices.contains(state.currentPathIndex))

            if state.segmentMethod == "manual" {
                Text("Click, or click-and-drag to draw freehand, around the vessel wall — then Use This Ring.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let error = state.errorMessage {
                Text(error).foregroundStyle(.red).font(.caption)
            }
        }
    }

    private var previewPane: some View {
        VStack(spacing: 6) {
            Text("Wall Preview")
                .font(.caption)
                .foregroundStyle(.secondary)
            SegmentPreview3DView(
                branch: branch, centerlineMM: state.seriesPointsMM,
                contoursByIndex: state.acceptedContoursMM[seriesKey] ?? [:],
                currentIndex: state.currentPathIndex,
                currentPointMM: state.seriesPointsMM.indices.contains(state.currentPathIndex)
                    ? state.seriesPointsMM[state.currentPathIndex] : nil,
                currentTangentMM: state.seriesTangentsMM.indices.contains(state.currentPathIndex)
                    ? state.seriesTangentsMM[state.currentPathIndex] : nil,
                currentRadiusMM: state.segmentRadiusMM,
                controller: previewController
            )
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(alignment: .topTrailing) {
                Scene3DControlsOverlay(controller: previewController)
            }
        }
    }

    private func step(_ delta: Int) async {
        state.currentPathIndex = min(max(state.currentPathIndex + delta, 0), state.seriesPointCount - 1)
        manualPoints = []
        // Restore the already-accepted result for this point, if any, so
        // stepping back to a point you already segmented still shows its
        // ring instead of a blank overlay.
        state.currentContourPx = state.acceptedContoursPx[seriesKey]?[state.currentPathIndex] ?? []
        state.lastFittedRadiusMM = state.acceptedRadiiMM[seriesKey]?[state.currentPathIndex]
    }

    private func loadSlice() async {
        isLoadingSlice = true
        defer { isLoadingSlice = false }
        image = await state.fetchSegmentSliceImage(side: side, branch: branch, wl: segWindowLevel, ww: segWindowWidth)
    }

    /// Converts a click location (pane-local, already accounting for
    /// zoom/pan via fitRect) to a pixel coordinate in the oblique slice's
    /// own [0, imageSize] space — the same space the server's own
    /// contour_px points use, so a manually-drawn ring submits in exactly
    /// the format segment_point's algorithmic output would.
    private func imagePixelPoint(from location: CGPoint, containerSize: CGSize, imageSize: CGSize) -> CGPoint? {
        let rect = fitRect(containerSize: containerSize, imageSize: imageSize, zoom: zoomScale, pan: panOffset)
        guard rect.width > 0, rect.height > 0 else { return nil }
        let nx = (location.x - rect.origin.x) / rect.width
        let ny = (location.y - rect.origin.y) / rect.height
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return nil }
        return CGPoint(x: nx * imageSize.width, y: ny * imageSize.height)
    }

    @ViewBuilder
    private func contourOverlay(containerSize: CGSize, imageSize: CGSize) -> some View {
        let rect = fitRect(containerSize: containerSize, imageSize: imageSize, zoom: zoomScale, pan: panOffset)
        let scale = rect.width / imageSize.width

        Path { path in
            guard let first = state.currentContourPx.first else { return }
            func point(_ rc: [Double]) -> CGPoint {
                CGPoint(x: rect.origin.x + rc[1] * scale, y: rect.origin.y + rc[0] * scale)
            }
            path.move(to: point(first))
            for rc in state.currentContourPx.dropFirst() {
                path.addLine(to: point(rc))
            }
            path.closeSubpath()
        }
        .stroke(branchColor(branch), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
    }

    /// The in-progress hand-drawn ring — open (not closed, since it isn't
    /// a ring yet) with a small dot at every placed vertex, so it's clear
    /// this is still being sketched.
    @ViewBuilder
    private func manualRingOverlay(containerSize: CGSize, imageSize: CGSize) -> some View {
        let rect = fitRect(containerSize: containerSize, imageSize: imageSize, zoom: zoomScale, pan: panOffset)
        let scale = rect.width / imageSize.width

        let screenPoint: (CGPoint) -> CGPoint = { p in
            CGPoint(x: rect.origin.x + p.x * scale, y: rect.origin.y + p.y * scale)
        }

        Path { path in
            guard let first = manualPoints.first else { return }
            path.move(to: screenPoint(first))
            for p in manualPoints.dropFirst() {
                path.addLine(to: screenPoint(p))
            }
        }
        .stroke(Color.white, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round, dash: [5, 3]))

        ForEach(Array(manualPoints.enumerated()), id: \.offset) { _, p in
            let sp = screenPoint(p)
            Circle()
                .fill(Color.white)
                .frame(width: 5, height: 5)
                .position(sp)
        }
    }
}
