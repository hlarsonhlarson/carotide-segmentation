import AppKit
import SwiftUI

/// Top-level case viewer: axial wide at top, coronal+sagittal as two
/// squares below it, the 3D path reconstruction as a narrow tall pane on
/// the right, every control button in horizontal rows below all of that,
/// and the contrast sliders at the very bottom.
struct SliceView: View {
    @EnvironmentObject var state: AppState
    @StateObject private var path3DController = Scene3DController()
    // Which side's surface to preview — captured at "Preview…" tap time
    // (not re-read from state.selectedSide while the sheet is open) so
    // flipping sides underneath an open preview can't retarget it mid-
    // view. Non-nil drives the sheet directly via Binding(get:set:), so
    // there's no separate "is the sheet open" Bool to keep in sync.
    @State private var previewSide: String?
    @State private var previewSource: SurfacePreviewSource = .buildSurface

    var body: some View {
        VStack(spacing: 12) {
            if let info = state.caseInfo {
                HStack {
                    Text(info.modality)
                        .font(Theme.labelFont(11, weight: .semibold))
                        .foregroundStyle(Theme.inkMuted)
                    Text(String(format: "%.2f × %.2f × %.2f mm", info.spacingMM[0], info.spacingMM[1], info.spacingMM[2]))
                        .font(Theme.readoutFont(11))
                        .foregroundStyle(Theme.inkMuted)
                    Spacer()
                }

                planesLayout
                    .frame(minHeight: 460)

                centerlineControls
                surfaceControls
                su2MeshControls
                boundaryConditionsControls
                su2RunControls

                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading) {
                        Text("Level (WL): \(Int(state.windowLevel))")
                            .font(Theme.labelFont(11))
                            .foregroundStyle(Theme.inkMuted)
                        Slider(value: $state.windowLevel, in: state.windowLevelRange)
                    }
                    VStack(alignment: .leading) {
                        Text("Width (WW): \(Int(state.windowWidth))")
                            .font(Theme.labelFont(11))
                            .foregroundStyle(Theme.inkMuted)
                        Slider(value: $state.windowWidth, in: state.windowWidthRange)
                    }
                    Button("Reset") { state.resetWindowToDefault() }
                        .buttonStyle(.themed)
                        .padding(.top, 14)
                }

                if let error = state.errorMessage {
                    Text(error).foregroundStyle(.red)
                }
            }
        }
        .padding()
        .background(Theme.background)
        .safeAreaInset(edge: .trailing) {
            if state.caseInfo != nil {
                CenterlinePointsPanel()
            }
        }
        .sheet(isPresented: $state.isSegmentingPath) {
            SegmentPathView(side: state.selectedSide, branch: state.selectedBranch)
                .environmentObject(state)
        }
        .sheet(isPresented: Binding(get: { previewSide != nil }, set: { if !$0 { previewSide = nil } })) {
            if let previewSide {
                SurfacePreviewView(side: previewSide, source: previewSource)
                    .environmentObject(state)
            }
        }
        .sheet(isPresented: $state.isBoundaryConditionsPresented) {
            BoundaryConditionsView(side: state.selectedSide)
                .environmentObject(state)
        }
        .sheet(isPresented: $state.isCFDResultsPresented) {
            if let caseId = state.caseInfo?.caseId {
                CFDResultsView(side: state.selectedSide, api: state.api, caseId: caseId)
                    .environmentObject(state)
            }
        }
    }

    private var centerlineControls: some View {
        HStack(spacing: 16) {
            Picker("Side", selection: $state.selectedSide) {
                Text("Left").tag("left")
                Text("Right").tag("right")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 140)
            .onChange(of: state.selectedSide) { newSide in
                Task { await state.refreshSU2StatusForSide(newSide) }
            }

            Picker("Branch", selection: $state.selectedBranch) {
                Text("CCA → ICA").tag("cca_ica")
                Text("CCA → ECA").tag("cca_eca")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 220)
            .tint(branchColor(state.selectedBranch))

            Toggle("Mark Centerline", isOn: $state.isMarkingCenterline)
                .toggleStyle(.button)
                .tint(state.isMarkingCenterline ? .red : nil)
                .onChange(of: state.isMarkingCenterline) { newValue in
                    if newValue {
                        state.isAutoTracingCenterline = false
                        state.autoTraceStartVoxel = nil
                    }
                }

            Toggle(isOn: $state.isAutoTracingCenterline) {
                if state.isAutoTracing {
                    Text("Tracing…")
                } else if state.isAutoTracingCenterline {
                    Text(state.autoTraceStartVoxel == nil ? "Auto-Trace: click start" : "Auto-Trace: click end")
                } else {
                    Text("Auto-Trace…")
                }
            }
            .toggleStyle(.button)
            .tint(state.isAutoTracingCenterline ? .red : nil)
            .disabled(state.isAutoTracing)
            .help("Click the branch's start point, then its end point — the path between is traced automatically. Best for easy, mostly-straight segments; review the result and add points by hand wherever a bend got cut short.")
            .onChange(of: state.isAutoTracingCenterline) { newValue in
                if newValue {
                    state.isMarkingCenterline = false
                } else {
                    state.autoTraceStartVoxel = nil
                }
            }

            Button("Clear Branch") {
                let side = state.selectedSide
                let branch = state.selectedBranch
                Task { await state.clearCenterline(side: side, branch: branch) }
            }
            .buttonStyle(.themed)

            Button("Segment Along Path…") {
                let side = state.selectedSide
                let branch = state.selectedBranch
                Task { await state.prepareSeries(side: side, branch: branch) }
            }
            .buttonStyle(.themedPrimary)
            .disabled((state.centerlinePaths[state.centerlineKey(side: state.selectedSide, branch: state.selectedBranch)] ?? []).count < 2)

            Spacer()

            if let path = state.centerlinePaths[state.centerlineKey(side: state.selectedSide, branch: state.selectedBranch)] {
                ReadoutLabel(value: "\(path.count)", caption: "points")
            }
        }
    }

    private var surfaceControls: some View {
        HStack(spacing: 16) {
            Button {
                let side = state.selectedSide
                Task { await state.buildSurface(side: side) }
            } label: {
                if state.isBuildingSurface {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Build Surface (\(state.selectedSide.capitalized))")
                }
            }
            .buttonStyle(.themedPrimary)
            .disabled(state.isBuildingSurface)

            if let result = state.lastBuildResult {
                if result.ok {
                    Label(
                        "\(result.nPoints ?? 0) pts, \(result.nBoundaryLoops ?? 0) open end(s)"
                            + (result.cleanTopology == true ? "" : " — check topology"),
                        systemImage: result.cleanTopology == true ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                    )
                    .font(Theme.labelFont(11, weight: .medium))
                    .foregroundStyle(result.cleanTopology == true ? Theme.ink : .orange)

                    Button("Preview…") {
                        previewSource = .buildSurface
                        previewSide = state.selectedSide
                    }
                    .buttonStyle(.themed)

                    Button("Save STL…") {
                        saveSurface()
                    }
                    .buttonStyle(.themed)
                } else {
                    Text(result.error ?? "Build failed")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            Spacer()
        }
    }

    private func saveSurface() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "stl")!]
        panel.nameFieldStringValue = "\(state.selectedSide)_carotid.stl"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let side = state.selectedSide
        Task { await state.saveSurfaceSTL(side: side, to: url) }
    }

    /// The actual CFD-ready deliverable: a volume mesh (.vtu) + tagged
    /// boundary surface (.vtp) via real TetGen meshing (see
    /// server/su2_export.py). Build Surface above stays as a fast visual
    /// sanity check; this is what's meant to actually feed a solver.
    private var su2MeshControls: some View {
        HStack(spacing: 16) {
            Button {
                let side = state.selectedSide
                Task { await state.startSU2Mesh(side: side) }
            } label: {
                if state.su2MeshState == "running" {
                    ProgressView().controlSize(.small)
                    Text("Meshing…")
                } else {
                    Text("Build CFD Mesh — \(state.selectedSide.capitalized)")
                }
            }
            .buttonStyle(.themedPrimary)
            .disabled(state.su2MeshState == "running")

            switch state.su2MeshState {
            case "done":
                Label("ready: \(state.su2MeshFaceMap?.count ?? 0) face(s)", systemImage: "checkmark.circle.fill")
                    .font(Theme.labelFont(11, weight: .medium))
                    .foregroundStyle(Theme.ink)
                Button("Preview…") {
                    previewSource = .su2Mesh
                    previewSide = state.selectedSide
                }
                .buttonStyle(.themed)
                Button("Save Mesh…") { saveSU2Mesh() }
                    .buttonStyle(.themed)
            case "error":
                Text(state.su2MeshError ?? "Mesh build failed")
                    .font(.caption)
                    .foregroundStyle(.red)
            default:
                EmptyView()
            }

            Spacer()
        }
    }

    /// The per-patient physiological inputs a future SU2 run needs (CCA
    /// inlet waveform, ICA/ECA flow split, outlet pressure) — see
    /// BoundaryConditionsView. Reachable regardless of mesh state, same
    /// "every step stays reachable" convention as the rest of this panel,
    /// though it's most relevant once a mesh actually exists.
    private var boundaryConditionsControls: some View {
        HStack(spacing: 16) {
            Button("Boundary Conditions…") {
                state.isBoundaryConditionsPresented = true
            }
            .buttonStyle(.themed)

            ReadoutLabel(
                value: String(format: "PSV %.0f / EDV %.0f cm/s", state.su2Params.psvCms, state.su2Params.edvCms),
                caption: String(format: "ICA %.0f%% · P %.0f/%.0f mmHg", state.su2Params.icaFraction * 100, state.su2Params.icaMapMmHg, state.su2Params.ecaMapMmHg),
                size: 11
            )

            Spacer()
        }
    }

    /// The actual CFD solve (server/su2_run.py) — a real mpirun+pysu2
    /// subprocess that can run for several minutes on a patient-size
    /// mesh (CONFIRMED on a real patient's own mesh: ~1s/time-step at
    /// 4 MPI processes, so a couple hundred steps is a few minutes, not
    /// hours), so this shows live step/ETA progress rather than just a
    /// spinner. Deliberately doesn't try to show or filter *results*
    /// here (WSS etc.) — that's its own, much bigger piece of work for
    /// later; this is just "run it and watch it go."
    private var su2RunControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 16) {
                Button {
                    let side = state.selectedSide
                    Task { await state.startSU2Run(side: side) }
                } label: {
                    if state.su2RunState == "running" || state.su2RunState == "cancelling" {
                        ProgressView().controlSize(.small)
                        Text("Running CFD…")
                    } else {
                        Text("Run CFD (\(state.selectedSide.capitalized))")
                    }
                }
                .buttonStyle(.themedPrimary)
                .disabled(
                    state.su2RunState == "running" || state.su2RunState == "cancelling"
                        || state.su2MeshState != "done"
                )

                if state.su2RunState == "running" {
                    Button("Cancel") {
                        let side = state.selectedSide
                        Task { await state.cancelSU2Run(side: side) }
                    }
                    .buttonStyle(.themed)
                }

                // su2PostprocessState == "done" also shows this even when
                // su2RunState == "idle" — a result restored from a saved
                // project (or still available after switching back from
                // another side, see AppState.refreshSU2StatusForSide) has
                // no live su2_run_jobs entry to make su2RunState anything
                // but idle, but the result itself is still genuinely
                // there to view.
                if state.su2RunState != "idle" || state.su2PostprocessState == "done" {
                    Button {
                        // A result already sitting in memory (this run's
                        // own, or one restored from a saved project — see
                        // AppState.openProject) just needs the viewer
                        // opened; re-extracting it from flow_*.vtu would
                        // fail outright for a restored result anyway,
                        // since there's no su2_run_jobs entry backing it
                        // after a project reopen.
                        if state.su2PostprocessState == "done" {
                            state.isCFDResultsPresented = true
                        } else {
                            let side = state.selectedSide
                            Task { await state.startSU2Postprocess(side: side) }
                        }
                    } label: {
                        if state.su2PostprocessState == "running" {
                            ProgressView().controlSize(.small)
                            Text("Extracting…")
                        } else {
                            // Works on a run that's still going or was
                            // cancelled too, not just "done" — whatever
                            // flow_*.vtu snapshots exist so far, see
                            // server/su2_postprocess.py.
                            Text("View Results…")
                        }
                    }
                    .buttonStyle(.themed)
                    .disabled(state.su2PostprocessState == "running")
                }

                switch state.su2RunState {
                case "done":
                    Label("CFD run complete", systemImage: "checkmark.circle.fill")
                        .font(Theme.labelFont(11, weight: .medium))
                        .foregroundStyle(Theme.ink)
                case "error":
                    Text(state.su2RunStatus?.error ?? "CFD run failed")
                        .font(.caption)
                        .foregroundStyle(.red)
                case "cancelled":
                    Label("Cancelled", systemImage: "xmark.circle")
                        .font(Theme.labelFont(11, weight: .medium))
                        .foregroundStyle(Theme.inkMuted)
                default:
                    if state.su2PostprocessState == "done" {
                        Label("Results available (from a previous run)", systemImage: "checkmark.circle")
                            .font(Theme.labelFont(11, weight: .medium))
                            .foregroundStyle(Theme.inkMuted)
                    } else if state.su2MeshState != "done" {
                        Text("Build the CFD mesh first")
                            .font(Theme.labelFont(11))
                            .foregroundStyle(Theme.inkMuted)
                    }
                }

                Spacer()
            }

            if let status = state.su2RunStatus, state.su2RunState == "running" || state.su2RunState == "cancelling" {
                VStack(alignment: .leading, spacing: 4) {
                    ThemedProgressBar(fraction: status.fractionComplete)
                    HStack {
                        Text(String(format: "step %d / %d", status.step ?? 0, status.totalSteps ?? 0))
                            .font(Theme.readoutFont(11))
                            .foregroundStyle(Theme.ink)
                        if let tSim = status.tSim, let cycleS = status.cycleS, cycleS > 0 {
                            Text(String(format: "t = %.2fs (%.1f cycles)", tSim, tSim / cycleS))
                                .font(Theme.readoutFont(11))
                                .foregroundStyle(Theme.inkMuted)
                        }
                        Spacer()
                        if let eta = status.etaSeconds {
                            Text("ETA \(formatDuration(eta))")
                                .font(Theme.readoutFont(11, weight: .bold))
                                .foregroundStyle(Theme.textBright)
                        }
                    }
                }
            }
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded()))
        let m = s / 60
        let r = s % 60
        return m > 0 ? "\(m)m \(r)s" : "\(r)s"
    }

    private func saveSU2Mesh() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Save Mesh Here"
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        let side = state.selectedSide
        let vtuURL = folder.appendingPathComponent("\(side)_carotid_mesh.vtu")
        let vtpURL = folder.appendingPathComponent("\(side)_carotid_mesh_surface.vtp")
        Task { await state.saveSU2Mesh(side: side, vtuURL: vtuURL, vtpURL: vtpURL) }
    }

    /// Which of the five workflow stages the selected side is currently
    /// at — the first one that isn't done yet, or Boundary Conditions
    /// once everything before it is. Purely informational (every step
    /// stays reachable
    /// regardless of this), just orientation for the WorkflowStepsPanel.
    /// Deliberately requires *both* paths (CCA→ICA and CCA→ECA), not just
    /// one of them, and for segmentation requires every one of a path's
    /// resampled points to actually have a contour — not just some of
    /// them — before that step's checkmark lights up.
    private var currentWorkflowStep: WorkflowStep {
        let branches = ["cca_ica", "cca_eca"]
        let hasPath = branches.allSatisfy { branch in
            (state.centerlinePaths[state.centerlineKey(side: state.selectedSide, branch: branch)] ?? []).count > 1
        }
        guard hasPath else { return .pathway }
        let hasSegmentation = branches.allSatisfy { branch in
            let key = state.centerlineKey(side: state.selectedSide, branch: branch)
            guard let total = state.seriesPointCountByKey[key], total > 0 else { return false }
            return (state.acceptedContoursMM[key]?.count ?? 0) >= total
        }
        guard hasSegmentation else { return .segmentation }
        guard state.lastBuildResult?.ok == true else { return .surface }
        guard state.su2MeshState == "done" else { return .meshing }
        guard state.su2RunState == "done" else { return .boundaryConditions }
        return .cfdComputation
    }

    /// Steps panel on the left (the free space there once the side
    /// length is height-limited, which is the usual case); axial wide at
    /// top (2:1 — "4x2"), coronal+sagittal as two squares below it in the
    /// middle; the 3D path reconstruction as a narrow tall pane on the
    /// right spanning the combined height of the other three. The points
    /// list sits further right still, via safeAreaInset.
    private var planesLayout: some View {
        GeometryReader { geo in
            let spacing: CGFloat = 10
            let stepsWidth: CGFloat = 320
            let pathWidthRatio: CGFloat = 0.62
            let sByWidth = (geo.size.width - 3 * spacing - stepsWidth) / (2 + pathWidthRatio)
            let sByHeight = (geo.size.height - spacing) / 2
            let side = max(min(sByWidth, sByHeight), 240)
            let pathWidth = side * pathWidthRatio

            HStack(spacing: spacing) {
                VStack(spacing: spacing) {
                    WorkflowStepsPanel(currentStep: currentWorkflowStep)
                    DebugPanel()
                }
                .frame(width: stepsWidth, height: 2 * side + spacing, alignment: .top)
                VStack(spacing: spacing) {
                    PlanePaneView(axis: .axial)
                        .frame(width: 2 * side, height: side)
                    HStack(spacing: spacing) {
                        PlanePaneView(axis: .coronal)
                            .frame(width: side, height: side)
                        PlanePaneView(axis: .sagittal)
                            .frame(width: side, height: side)
                    }
                }
                path3DPane
                    .frame(width: pathWidth, height: 2 * side + spacing)
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topTrailing)
        }
    }

    private var path3DPane: some View {
        VStack(spacing: 4) {
            Text("3D PATH")
                .font(Theme.labelFont(10, weight: .bold))
                .foregroundStyle(Theme.inkMuted)
            let paths = ["cca_ica", "cca_eca"].compactMap { branch -> (branch: String, path: [[Int]], controlPoints: [[Int]])? in
                let key = state.centerlineKey(side: state.selectedSide, branch: branch)
                guard let pts = state.centerlinePaths[key], pts.count > 1 else { return nil }
                return (branch, pts, state.controlPoints[key] ?? [])
            }
            if paths.isEmpty {
                Spacer()
                Text("Mark a path to preview it in 3D")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            } else if let info = state.caseInfo {
                let wallContoursByBranch = Dictionary(uniqueKeysWithValues: ["cca_ica", "cca_eca"].map { branch in
                    (branch, state.acceptedContoursMM[state.centerlineKey(side: state.selectedSide, branch: branch)] ?? [:])
                })
                PathPreview3DView(
                    paths: paths, spacingMM: info.spacingMM, originMM: info.originMM,
                    wallContoursByBranch: wallContoursByBranch,
                    highlightBranch: state.reorderingPointKey?.components(separatedBy: ":").last,
                    highlightIndex: state.reorderingPointIndex,
                    controller: path3DController
                )
                .overlay(alignment: .topTrailing) {
                    Scene3DControlsOverlay(controller: path3DController)
                }
            }
        }
        .frame(minWidth: 220, maxWidth: .infinity, minHeight: 220, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 6).fill(Theme.panel))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
