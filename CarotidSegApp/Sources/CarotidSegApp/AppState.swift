import AppKit
import Foundation

enum PlaneAxis: String, CaseIterable, Identifiable, Hashable {
    case axial, coronal, sagittal
    var id: String { rawValue }
    var label: String {
        switch self {
        case .axial: return "Axial"
        case .coronal: return "Coronal"
        case .sagittal: return "Sagittal"
        }
    }
    /// Index into a (z, y, x) shape array for this plane's "through" axis.
    var shapeIndex: Int {
        switch self {
        case .axial: return 0
        case .coronal: return 1
        case .sagittal: return 2
        }
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var caseInfo: CaseInfo?
    @Published var axialIndex: Int = 0
    @Published var coronalIndex: Int = 0
    @Published var sagittalIndex: Int = 0
    // Clamped on every write (not just read) so a Slider bound directly to
    // these can never observe a value outside its `in:` range — SwiftUI's
    // Slider crashes with a hard precondition failure otherwise (confirmed
    // via a real crash report). Clamping only on read (via a wrapping
    // Binding) turned out unreliable for external writes like the Reset
    // button — this is the more standard, directly-bindable pattern.
    @Published var windowLevel: Double = 0 {
        didSet {
            let r = windowLevelRange
            if windowLevel < r.lowerBound { windowLevel = r.lowerBound }
            else if windowLevel > r.upperBound { windowLevel = r.upperBound }
        }
    }
    @Published var windowWidth: Double = 1 {
        didSet {
            let r = windowWidthRange
            if windowWidth < r.lowerBound { windowWidth = r.lowerBound }
            else if windowWidth > r.upperBound { windowWidth = r.upperBound }
        }
    }
    @Published var errorMessage: String?
    @Published var isBusy: Bool = false

    // Manual centerline tracing (the "auto-dotyazhka" phase): which branch
    // is currently being edited, whether clicks mark points instead of just
    // moving the cursor, and the traced paths themselves — keyed exactly
    // like the server's response ("side:branch" -> [[z,y,x], ...]).
    @Published var selectedSide: String = "right"
    // Only two paths exist: each runs the *full* length from the common
    // carotid trunk through the bifurcation to its own tip (not three
    // disjoint cca/ica/eca segments stitched only at a shared point) —
    // segmenting one continuous path keeps the tangent/contour sequence
    // smooth through the bifurcation instead of kinking where three
    // independently-marked segments used to meet.
    @Published var selectedBranch: String = "cca_ica"
    @Published var isMarkingCenterline: Bool = false
    // Auto-Trace mode: click exactly two points (branch start, then tip)
    // and the whole path between them is traced automatically — see
    // AppState.handleAutoTraceClick and server/app.py's /auto_trace.
    // Mutually exclusive with isMarkingCenterline (entering one turns the
    // other off, so a click can never be ambiguous about which mode it's
    // for). autoTraceStartVoxel holds the first click until the second
    // one arrives and actually fires the request.
    @Published var isAutoTracingCenterline: Bool = false
    @Published var autoTraceStartVoxel: [Int]?
    @Published var isAutoTracing: Bool = false
    @Published var centerlinePaths: [String: [[Int]]] = [:]
    // The sparse points the user actually clicked/placed, in the same
    // "side:branch" keying — separate from centerlinePaths (the dense
    // stitched route) so an individual placed point can be shown, dragged,
    // or deleted.
    @Published var controlPoints: [String: [[Int]]] = [:]
    // While dragging an existing control point: which (side,branch,index)
    // and its live (uncommitted) position, for immediate visual feedback
    // without hitting the network on every drag event.
    @Published var draggingPointKey: String?
    @Published var draggingPointIndex: Int?
    @Published var draggingPointVoxel: [Int]?

    // Which point is being dragged within the side-panel's reorder list
    // right now (as opposed to draggingPoint* above, which is for dragging
    // a marker directly on the 2D slice) — drives the red highlight on
    // that point's 3D sphere, and the drop-target outline in the list,
    // while the drag is in flight.
    @Published var reorderingPointKey: String?
    @Published var reorderingPointIndex: Int?
    @Published var reorderDropTargetKey: String?
    @Published var reorderDropTargetIndex: Int?

    // SimVascular-style per-point segmentation (the Series/ContourGroup
    // step): resample the marked path, step through it, run Threshold or
    // Level Set at each point against the real oblique slice.
    @Published var isSegmentingPath: Bool = false
    @Published var isSeedingCircles: Bool = false
    @Published var seriesPointCount: Int = 0
    // Same n_points as seriesPointCount, but kept per "side:branch" and
    // never reset when the segmentation sheet closes — needed so the
    // Workflow panel can tell "every point on this path has a contour"
    // (acceptedContoursMM[key].count >= this) apart from "nothing's been
    // prepared yet" for a branch that currently isn't being edited.
    @Published var seriesPointCountByKey: [String: Int] = [:]
    @Published var currentPathIndex: Int = 0
    @Published var segmentMethod: String = "threshold"
    @Published var segmentRadiusMM: Double = 2.5
    @Published var currentContourPx: [[Double]] = []
    @Published var lastFittedRadiusMM: Double?
    @Published var acceptedIndices: Set<Int> = []
    // The resampled Series centerline (world-mm, [z,y,x] per point) for
    // the branch currently being edited in SegmentPathView.
    @Published var seriesPointsMM: [[Double]] = []
    @Published var seriesTangentsMM: [[Double]] = []
    // Every accepted point's 3D contour loop (world-mm), keyed by
    // "side:branch" then path index — *not* reset when the segmentation
    // sheet closes, so the wall preview it built keeps showing in the
    // main 3D path view afterward. Pixel-space contours + fitted radii are
    // kept the same way, restored into currentContourPx/lastFittedRadiusMM
    // when stepping back to an already-segmented point so going back
    // doesn't make its result (and the overlay showing it) disappear.
    @Published var acceptedContoursMM: [String: [Int: [[Double]]]] = [:]
    @Published var acceptedContoursPx: [String: [Int: [[Double]]]] = [:]
    @Published var acceptedRadiiMM: [String: [Int: Double]] = [:]

    // Final surface build (rasterize segmented Series -> mesh-ready STL).
    @Published var isBuildingSurface: Bool = false
    @Published var lastBuildResult: APIClient.BuildSurfaceResult?
    // "idle" | "running" | "done" | "error" — the SU2-ready volume+surface
    // mesh job (see AppState.startSU2Mesh).
    @Published var su2MeshState: String = "idle"
    @Published var su2MeshFaceMap: [String: String]?
    @Published var su2MeshError: String?
    // Raw status JSON from the last poll — live stage/stage_info while
    // running, final diag (surface + volume counts, per-face triangle
    // counts) once done. DebugPanel shows this directly.
    @Published var su2MeshStatusJSON: String?

    // Boundary Conditions step: the per-side physiological parameters a
    // future SU2 run is driven by (see server/app.py's SU2_PARAM_DEFAULTS
    // for the same defaults this starts at before anything's fetched).
    @Published var isBoundaryConditionsPresented: Bool = false
    @Published var su2Params = APIClient.SU2Params(
        psvCms: 85.0, edvCms: 30.0, cardiacCycleS: 0.8, icaFraction: 0.65,
        icaMapMmHg: 93.0, ecaMapMmHg: 93.0
    )
    @Published var isLoadingSU2Params: Bool = false

    // The actual SU2 pulsatile CFD solve (not just the mesh) — a real
    // mpirun+pysu2 subprocess that can run for several minutes, so this
    // tracks live step/ETA progress for the UI rather than just
    // idle/running/done. See AppState.startSU2Run.
    @Published var su2RunState: String = "idle"
    @Published var su2RunStatus: APIClient.SU2RunStatus?

    // Post-processed CFD results (TAWSS/OSI/RRT/pressure on the wall) —
    // built from a finished (or still-running/cancelled) su2_run's own
    // flow_*.vtu snapshots. See AppState.startSU2Postprocess /
    // CFDResultsView, which does its own fetch of the actual geometry +
    // field payload once this reaches "done" — this just tracks whether
    // a result is ready to view.
    @Published var su2PostprocessState: String = "idle"
    @Published var isCFDResultsPresented: Bool = false

    // Folder this patient's work is persisted into — set once per case,
    // right after loading it (see CaseLoaderView), and from then on every
    // workflow step's own Save button writes there (see save*ToProject
    // below). nil until the user picks one (or declines to).
    @Published var projectDir: String?

    let api = APIClient()

    func loadCase(path: String, seriesUID: String?) async {
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        do {
            let info = try await api.createCase(path: path, seriesUID: seriesUID)
            caseInfo = info
            axialIndex = info.shape[PlaneAxis.axial.shapeIndex] / 2
            coronalIndex = info.shape[PlaneAxis.coronal.shapeIndex] / 2
            sagittalIndex = info.shape[PlaneAxis.sagittal.shapeIndex] / 2
            windowLevel = info.defaultWindow.wl
            windowWidth = info.defaultWindow.ww
            centerlinePaths = [:]
            controlPoints = [:]
            projectDir = nil
            await refreshCenterlines()
            await chooseProjectFolder()
        } catch {
            errorMessage = "Failed to load case: \(error.localizedDescription)"
        }
    }

    /// Reopens a project folder saved by the workflow panel's own Save
    /// buttons: reloads the same DICOM series, then restores whatever
    /// centerline/segmentation data that folder has into both the
    /// backend's in-memory CaseState (done server-side, see
    /// server/app.py's /projects/open) and this client's own published
    /// dictionaries (acceptedContoursMM/acceptedContoursPx/
    /// acceptedRadiiMM/seriesPointCountByKey), which the restore call
    /// itself doesn't touch since it only returns a summary of what got
    /// restored, not the per-point data itself — fetched here the same
    /// way SegmentPathView would for an already-segmented branch.
    /// Surface/mesh deliberately aren't restored the same way — see that
    /// endpoint's own docstring for why — so after reopening, Build
    /// Surface / Build SU2 Mesh need one click each to regenerate them.
    func openProject(path: String) async {
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        do {
            let result = try await api.openProject(path: path)
            let info = result.caseInfo
            caseInfo = info
            axialIndex = info.shape[PlaneAxis.axial.shapeIndex] / 2
            coronalIndex = info.shape[PlaneAxis.coronal.shapeIndex] / 2
            sagittalIndex = info.shape[PlaneAxis.sagittal.shapeIndex] / 2
            windowLevel = info.defaultWindow.wl
            windowWidth = info.defaultWindow.ww
            centerlinePaths = [:]
            controlPoints = [:]
            acceptedContoursMM = [:]
            acceptedContoursPx = [:]
            acceptedRadiiMM = [:]
            seriesPointCountByKey = [:]
            projectDir = result.projectDir
            // su2RunState/su2PostprocessState aren't tracked per side (a
            // pre-existing simplification — see su2MeshState, same deal),
            // so this only reflects whatever the *currently selected*
            // side restored. A saved CFD result is restored straight into
            // the backend's su2_postprocess_jobs (see server/app.py's
            // /projects/open) — re-deriving it would mean re-running the
            // actual solve — so "done" here just unlocks the already-
            // restored "View Results…" button, it doesn't claim a run
            // happened in this session.
            if result.restored[selectedSide]?.cfdResults == true {
                su2RunState = "done"
                su2PostprocessState = "done"
            } else {
                su2RunState = "idle"
                su2PostprocessState = "idle"
            }
            su2RunStatus = nil
            await refreshCenterlines()

            guard let caseId = caseInfo?.caseId else { return }
            for (side, sideInfo) in result.restored {
                for branch in sideInfo.segmentation {
                    let key = centerlineKey(side: side, branch: branch)
                    do {
                        let points = try await api.fetchSegmentationSeries(caseId: caseId, side: side, branch: branch)
                        seriesPointCountByKey[key] = points.count
                        var contoursMM: [Int: [[Double]]] = [:]
                        var contoursPx: [Int: [[Double]]] = [:]
                        var radii: [Int: Double] = [:]
                        for (idx, p) in points.enumerated() {
                            if let mm = p.contour_mm { contoursMM[idx] = mm }
                            if let px = p.contour_px { contoursPx[idx] = px }
                            if let r = p.radius_mm { radii[idx] = r }
                        }
                        acceptedContoursMM[key] = contoursMM
                        acceptedContoursPx[key] = contoursPx
                        acceptedRadiiMM[key] = radii
                    } catch {
                        // Non-fatal — that one branch's contours just
                        // won't preview/count as accepted; the rest of
                        // the restore still succeeds.
                    }
                }
            }
        } catch {
            errorMessage = "Failed to open project: \(error.localizedDescription)"
        }
    }

    /// Prompts for the folder this patient's work (centerline, surface,
    /// mesh — every workflow step's own Save button) gets persisted into.
    /// Called once automatically right after a case loads; also callable
    /// again later (e.g. the first time a Save button is pressed and
    /// projectDir is still nil, because the user dismissed this prompt)
    /// without needing to reload the case.
    @discardableResult
    func chooseProjectFolder() async -> Bool {
        guard let caseId = caseInfo?.caseId else { return false }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose Project Folder"
        panel.message = "Choose (or create) a folder to save this patient's centerline, surface, and mesh into."
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        do {
            projectDir = try await api.setProjectDir(caseId: caseId, path: url.path)
            return true
        } catch {
            errorMessage = "Failed to set project folder: \(error.localizedDescription)"
            return false
        }
    }

    /// Shared by every per-step Save button: make sure a project folder
    /// is set (prompting for one if the user dismissed the automatic
    /// prompt at load time) before actually asking the backend to write
    /// anything into it.
    private func ensureProjectFolder() async -> Bool {
        if projectDir != nil { return true }
        return await chooseProjectFolder()
    }

    func saveCenterlineToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveCenterlineToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save centerline: \(error.localizedDescription)"
        }
    }

    func saveSegmentationToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveSegmentationToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save segmentation: \(error.localizedDescription)"
        }
    }

    func saveSurfaceToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveSurfaceToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save surface: \(error.localizedDescription)"
        }
    }

    func saveMeshToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveMeshToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save mesh: \(error.localizedDescription)"
        }
    }

    func saveSU2ParamsToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveSU2ParamsToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save boundary conditions: \(error.localizedDescription)"
        }
    }

    func saveCFDResultsToProject(side: String) async {
        guard let caseId = caseInfo?.caseId, await ensureProjectFolder() else { return }
        do {
            try await api.saveCFDResultsToProject(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to save CFD results: \(error.localizedDescription)"
        }
    }

    /// Fetches this side's stored boundary-condition parameters (or the
    /// backend's generic defaults if none were set yet) — called when the
    /// Boundary Conditions sheet opens, so it always reflects this side's
    /// own last-saved values rather than whatever side was open before.
    func loadSU2Params(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        isLoadingSU2Params = true
        defer { isLoadingSU2Params = false }
        do {
            su2Params = try await api.fetchSU2Params(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to load boundary conditions: \(error.localizedDescription)"
        }
    }

    /// Pushes the sheet's current (in-memory, already-edited) values to
    /// the backend — called on every field edit (cheap, local JSON) so
    /// the in-memory case always has the latest values even if the user
    /// closes the sheet without an explicit Save, same spirit as every
    /// other step here not gating backend state on a modal "OK" button.
    func updateSU2Params() async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            su2Params = try await api.postSU2Params(caseId: caseId, side: selectedSide, params: su2Params)
        } catch {
            errorMessage = "Failed to update boundary conditions: \(error.localizedDescription)"
        }
    }

    func refreshCenterlines() async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            centerlinePaths = try await api.fetchCenterlines(caseId: caseId)
            controlPoints = try await api.fetchControlPoints(caseId: caseId)
        } catch {
            // Non-fatal — the viewer still works without existing paths.
        }
    }

    func index(for axis: PlaneAxis) -> Int {
        switch axis {
        case .axial: return axialIndex
        case .coronal: return coronalIndex
        case .sagittal: return sagittalIndex
        }
    }

    func setIndex(_ value: Int, for axis: PlaneAxis) {
        let clamped = min(max(value, 0), max(sliceCount(for: axis) - 1, 0))
        switch axis {
        case .axial: axialIndex = clamped
        case .coronal: coronalIndex = clamped
        case .sagittal: sagittalIndex = clamped
        }
    }

    func sliceCount(for axis: PlaneAxis) -> Int {
        caseInfo?.shape[axis.shapeIndex] ?? 1
    }

    /// Which voxel a normalized [0,1] fraction along an axis of `count`
    /// voxels falls in — the rendered image is `count` equal-width pixel
    /// bucket(s) spanning the pane, so a click's own voxel is just "which
    /// bucket is the cursor over" (floor(frac * count)), matching how the
    /// bitmap is actually rasterized onto the pane. NOTE: this is
    /// deliberately NOT count-1-based "pick the nearest of count sample
    /// points" — that was tried and made things measurably *worse*
    /// (reported directly), reverted back to this simpler, original form.
    /// See indexToCenterFraction below for the matching inverse used when
    /// drawing a voxel back onto the pane.
    private func indexFromFraction(_ frac: Double, _ count: Int) -> Int {
        guard count > 0 else { return 0 }
        return min(max(Int(frac * Double(count)), 0), count - 1)
    }

    /// Inverse of indexFromFraction: the normalized [0,1] position of the
    /// *center* of voxel `v`'s own pixel bucket — used by
    /// crosshairNormalized/normalizedProjection to draw a voxel back onto
    /// the pane so it lands inside the same bucket a click there would
    /// have produced, not just near it.
    private func indexToCenterFraction(_ v: Int, _ count: Int) -> Double {
        guard count > 0 else { return 0.5 }
        return (Double(v) + 0.5) / Double(count)
    }

    /// axialIndex/coronalIndex/sagittalIndex already form one shared 3D
    /// cursor (Z,Y,X) — clicking in one plane just needs to move the other
    /// two indices to match the clicked position, the way any MPR viewer
    /// (SimVascular included) keeps all three views in sync. `nx`/`ny` are
    /// normalized [0,1] across the displayed plane image, (0,0) = top-left.
    /// Coronal/sagittal are displayed row-flipped (server flips them so
    /// superior renders at the top — see server/app.py get_slice), so their
    /// row axis runs z=max at the top, z=0 at the bottom; invert that back
    /// to get a true z index.
    func updateCursorFromClick(axis: PlaneAxis, nx: Double, ny: Double) {
        guard let shape = caseInfo?.shape else { return }
        let nx = min(max(nx, 0), 1)
        let ny = min(max(ny, 0), 1)
        switch axis {
        case .axial:
            setIndex(indexFromFraction(nx, shape[2]), for: .sagittal)
            setIndex(indexFromFraction(ny, shape[1]), for: .coronal)
        case .coronal:
            setIndex(indexFromFraction(nx, shape[2]), for: .sagittal)
            let zFromTop = indexFromFraction(ny, shape[0])
            setIndex(shape[0] - 1 - zFromTop, for: .axial)
        case .sagittal:
            setIndex(indexFromFraction(nx, shape[1]), for: .coronal)
            let zFromTop = indexFromFraction(ny, shape[0])
            setIndex(shape[0] - 1 - zFromTop, for: .axial)
        }
    }

    /// Inverse of updateCursorFromClick: where the *other* two indices
    /// currently project onto this plane, normalized [0,1] — used to draw
    /// the crosshair. Returns nil if there's no loaded case.
    func crosshairNormalized(for axis: PlaneAxis) -> (x: Double, y: Double)? {
        guard let shape = caseInfo?.shape else { return nil }
        func frac(_ v: Int, _ count: Int) -> Double { indexToCenterFraction(v, count) }
        switch axis {
        case .axial:
            return (frac(sagittalIndex, shape[2]), frac(coronalIndex, shape[1]))
        case .coronal:
            let zFromTop = shape[0] - 1 - axialIndex
            return (frac(sagittalIndex, shape[2]), frac(zFromTop, shape[0]))
        case .sagittal:
            let zFromTop = shape[0] - 1 - axialIndex
            return (frac(coronalIndex, shape[1]), frac(zFromTop, shape[0]))
        }
    }

    /// Full 3D voxel [z,y,x] for a click at normalized (nx,ny) on a given
    /// pane — same per-axis geometry as updateCursorFromClick, but this
    /// plane only tells you 2 of the 3 coordinates; the third comes from
    /// whichever index is already current for the axis perpendicular to it
    /// (the depth the user is currently scrolled to on that plane).
    func voxelFromClick(axis: PlaneAxis, nx: Double, ny: Double) -> [Int]? {
        guard let shape = caseInfo?.shape else { return nil }
        let nx = min(max(nx, 0), 1)
        let ny = min(max(ny, 0), 1)
        switch axis {
        case .axial:
            let x = indexFromFraction(nx, shape[2])
            let y = indexFromFraction(ny, shape[1])
            return [axialIndex, y, x]
        case .coronal:
            let x = indexFromFraction(nx, shape[2])
            let zFromTop = indexFromFraction(ny, shape[0])
            return [shape[0] - 1 - zFromTop, coronalIndex, x]
        case .sagittal:
            let y = indexFromFraction(nx, shape[1])
            let zFromTop = indexFromFraction(ny, shape[0])
            return [shape[0] - 1 - zFromTop, y, sagittalIndex]
        }
    }

    /// Where an arbitrary stored voxel point projects onto a given plane,
    /// normalized [0,1] — generalizes crosshairNormalized (which only
    /// projects the *current cursor*) to any traced centerline point, for
    /// drawing the path overlay.
    func normalizedProjection(axis: PlaneAxis, voxel: [Int]) -> (x: Double, y: Double)? {
        guard let shape = caseInfo?.shape else { return nil }
        func frac(_ v: Int, _ count: Int) -> Double { indexToCenterFraction(v, count) }
        switch axis {
        case .axial:
            return (frac(voxel[2], shape[2]), frac(voxel[1], shape[1]))
        case .coronal:
            let zFromTop = shape[0] - 1 - voxel[0]
            return (frac(voxel[2], shape[2]), frac(zFromTop, shape[0]))
        case .sagittal:
            let zFromTop = shape[0] - 1 - voxel[0]
            return (frac(voxel[1], shape[1]), frac(zFromTop, shape[0]))
        }
    }

    func centerlineKey(side: String, branch: String) -> String { "\(side):\(branch)" }

    /// Any edit to a branch's path invalidates whatever segmentation was
    /// already fit to it — mirrors server/app.py's own
    /// _invalidate_segmentation (same bug it fixes: re-tracing a branch
    /// after it was already segmented left the 3D Path pane drawing the
    /// *new* centerline next to the *old* segmentation's own wall
    /// contours, built for a path that no longer exists). Already the
    /// established pattern in clearCenterline below; every other path-
    /// mutating call needs it too.
    private func invalidateSegmentation(key: String) {
        seriesPointCountByKey[key] = nil
        acceptedContoursMM[key] = nil
        acceptedContoursPx[key] = nil
        acceptedRadiiMM[key] = nil
    }

    func addCenterlinePoint(side: String, branch: String, voxel: [Int]) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            let result = try await api.addCenterlinePoint(caseId: caseId, side: side, branch: branch, point: voxel)
            let key = centerlineKey(side: side, branch: branch)
            centerlinePaths[key] = result.path
            controlPoints[key] = result.controlPoints
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Failed to add centerline point: \(error.localizedDescription)"
        }
    }

    /// Called for every click while isAutoTracingCenterline is on — the
    /// first click is just remembered (nothing to send yet), the second
    /// one fires the actual auto-trace request and then exits the mode
    /// (one trace per activation, same "click twice and you're done"
    /// shape as the rest of this tool rather than staying armed).
    func handleAutoTraceClick(side: String, branch: String, voxel: [Int]) async {
        guard let start = autoTraceStartVoxel else {
            autoTraceStartVoxel = voxel
            return
        }
        guard let caseId = caseInfo?.caseId else { return }
        isAutoTracing = true
        defer {
            isAutoTracing = false
            autoTraceStartVoxel = nil
            isAutoTracingCenterline = false
        }
        do {
            let result = try await api.autoTraceCenterline(caseId: caseId, side: side, branch: branch, start: start, end: voxel)
            let key = centerlineKey(side: side, branch: branch)
            centerlinePaths[key] = result.path
            controlPoints[key] = result.controlPoints
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Auto-trace failed: \(error.localizedDescription)"
        }
    }

    func moveCenterlinePoint(side: String, branch: String, index: Int, voxel: [Int]) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            let result = try await api.moveCenterlinePoint(caseId: caseId, side: side, branch: branch, index: index, point: voxel)
            let key = centerlineKey(side: side, branch: branch)
            centerlinePaths[key] = result.path
            controlPoints[key] = result.controlPoints
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Failed to move centerline point: \(error.localizedDescription)"
        }
    }

    func deleteCenterlinePoint(side: String, branch: String, index: Int) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            let result = try await api.deleteCenterlinePoint(caseId: caseId, side: side, branch: branch, index: index)
            let key = centerlineKey(side: side, branch: branch)
            centerlinePaths[key] = result.path
            controlPoints[key] = result.controlPoints
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Failed to delete centerline point: \(error.localizedDescription)"
        }
    }

    func reorderCenterlinePoint(side: String, branch: String, fromIndex: Int, toIndex: Int) async {
        guard let caseId = caseInfo?.caseId else { return }
        let key = centerlineKey(side: side, branch: branch)
        // Optimistic local reorder first — the list-reorder drag needs to
        // feel immediate, and re-stitching the path server-side (geodesic
        // search between every affected pair) is not instant.
        if var control = controlPoints[key], fromIndex >= 0, fromIndex < control.count,
           toIndex >= 0, toIndex < control.count {
            let moved = control.remove(at: fromIndex)
            control.insert(moved, at: toIndex)
            controlPoints[key] = control
        }
        do {
            let result = try await api.reorderCenterlinePoint(
                caseId: caseId, side: side, branch: branch, fromIndex: fromIndex, toIndex: toIndex
            )
            centerlinePaths[key] = result.path
            controlPoints[key] = result.controlPoints
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Failed to reorder centerline point: \(error.localizedDescription)"
        }
    }

    func clearCenterline(side: String, branch: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            try await api.clearCenterline(caseId: caseId, side: side, branch: branch)
            let key = centerlineKey(side: side, branch: branch)
            centerlinePaths[key] = nil
            controlPoints[key] = nil
            invalidateSegmentation(key: key)
        } catch {
            errorMessage = "Failed to clear centerline: \(error.localizedDescription)"
        }
    }

    func prepareSeries(side: String, branch: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        let key = centerlineKey(side: side, branch: branch)
        do {
            let result = try await api.prepareSeries(caseId: caseId, side: side, branch: branch)
            seriesPointCount = result.nPoints
            seriesPointCountByKey[key] = result.nPoints
            seriesPointsMM = result.pointsMM
            seriesTangentsMM = result.tangentsMM
            currentPathIndex = 0
            currentContourPx = []
            lastFittedRadiusMM = nil
            acceptedIndices = []
            acceptedContoursMM[key] = [:]
            acceptedContoursPx[key] = [:]
            acceptedRadiiMM[key] = [:]
            isSegmentingPath = seriesPointCount > 0
            if seriesPointCount == 0 {
                errorMessage = "Path has no resampled points — mark a longer path first."
            } else {
                await seedCircularContours(side: side, branch: branch)
            }
        } catch {
            errorMessage = "Failed to prepare series: \(error.localizedDescription)"
        }
    }

    /// Immediately after preparing a Series, seeds every point by really
    /// running Level Set — keeping that result when it looks like a
    /// trustworthy, reasonably round cross-section, and only falling back
    /// to a plain circle at the current default radius when it doesn't
    /// (a leak, a failed detection) — see the server's "auto" method and
    /// _contour_compactness. The result is a complete wall preview right
    /// away, correct wherever Level Set actually worked, rather than
    /// *every* point being a fake circle. Every point is still free to be
    /// re-segmented individually with Level Set / a manually-drawn ring
    /// afterward.
    func seedCircularContours(side: String, branch: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        let key = centerlineKey(side: side, branch: branch)
        let radius = segmentRadiusMM
        let n = seriesPointCount
        let apiClient = api
        guard n > 0 else { return }
        isSeedingCircles = true
        defer { isSeedingCircles = false }
        let results: [(Int, APIClient.SegmentPointResult?)] = await withTaskGroup(of: (Int, APIClient.SegmentPointResult?).self) { group in
            for idx in 0..<n {
                group.addTask {
                    do {
                        let result = try await apiClient.segmentPoint(
                            caseId: caseId, side: side, branch: branch, index: idx,
                            method: "auto", radiusMM: radius
                        )
                        return (idx, result)
                    } catch {
                        return (idx, nil)
                    }
                }
            }
            var collected: [(Int, APIClient.SegmentPointResult?)] = []
            for await r in group { collected.append(r) }
            return collected
        }
        // Apply every result through a handful of *batched* writes rather
        // than one @Published mutation per point — with 50-100+ points,
        // writing acceptedIndices/acceptedContoursMM/etc. separately for
        // each one fires that many SwiftUI re-renders back to back (every
        // view reading AppState, including the main window's panes,
        // re-evaluates on each one), which was a real, confirmed source
        // of the main thread falling behind and unrelated UI (an axial
        // slider mid-scroll) stuttering or misbehaving.
        var newIndices = acceptedIndices
        var newContoursMM = acceptedContoursMM[key] ?? [:]
        var newContoursPx = acceptedContoursPx[key] ?? [:]
        var newRadii = acceptedRadiiMM[key] ?? [:]
        for (idx, result) in results.sorted(by: { $0.0 < $1.0 }) {
            guard let result else { continue }
            newIndices.insert(idx)
            newContoursMM[idx] = result.contourMM
            newContoursPx[idx] = result.contourPx
            if let fitted = result.fittedRadiusMM { newRadii[idx] = fitted }
            if idx == currentPathIndex {
                currentContourPx = result.contourPx
                lastFittedRadiusMM = result.fittedRadiusMM
            }
        }
        acceptedIndices = newIndices
        acceptedContoursMM[key] = newContoursMM
        acceptedContoursPx[key] = newContoursPx
        acceptedRadiiMM[key] = newRadii
    }

    func fetchSegmentSliceImage(side: String, branch: String, wl: Double? = nil, ww: Double? = nil) async -> NSImage? {
        guard let caseId = caseInfo?.caseId else { return nil }
        do {
            return try await api.fetchSegmentSlice(
                caseId: caseId, side: side, branch: branch, index: currentPathIndex,
                wl: wl ?? windowLevel, ww: ww ?? windowWidth
            )
        } catch is CancellationError {
            return nil
        } catch let urlError as URLError where urlError.code == .cancelled {
            return nil
        } catch {
            errorMessage = "Failed to fetch oblique slice: \(error.localizedDescription)"
            return nil
        }
    }

    func segmentCurrentPoint(side: String, branch: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        do {
            let result = try await api.segmentPoint(
                caseId: caseId, side: side, branch: branch, index: currentPathIndex,
                method: segmentMethod, radiusMM: segmentRadiusMM
            )
            currentContourPx = result.contourPx
            lastFittedRadiusMM = result.fittedRadiusMM
            if let fitted = result.fittedRadiusMM { segmentRadiusMM = fitted }
            acceptedIndices.insert(currentPathIndex)
            applySegmentResult(key: centerlineKey(side: side, branch: branch), index: currentPathIndex, result: result)
        } catch {
            errorMessage = "Segmentation failed: \(error.localizedDescription)"
        }
    }

    /// A ring the user drew by hand on the oblique slice (clicking vertices
    /// directly) instead of running Threshold/Level Set — for the cases
    /// where automatic detection picks up the wrong structure and it's
    /// faster to just trace the wall yourself. `contourPx` is in the same
    /// [row, col] pixel space the server's own contours use.
    func submitManualRing(side: String, branch: String, contourPx: [[Double]]) async {
        guard let caseId = caseInfo?.caseId else { return }
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        do {
            let result = try await api.segmentPointManual(
                caseId: caseId, side: side, branch: branch, index: currentPathIndex, contourPx: contourPx
            )
            currentContourPx = result.contourPx
            lastFittedRadiusMM = result.fittedRadiusMM
            if let fitted = result.fittedRadiusMM { segmentRadiusMM = fitted }
            acceptedIndices.insert(currentPathIndex)
            applySegmentResult(key: centerlineKey(side: side, branch: branch), index: currentPathIndex, result: result)
        } catch {
            errorMessage = "Manual ring failed: \(error.localizedDescription)"
        }
    }

    /// Clears the current point back to "not segmented" — for a result
    /// that's plain wrong (a leak, a bad auto-seed) where a gap in the
    /// wall preview is more honest than leaving a wrong ring in place.
    func deleteCurrentSegmentation(side: String, branch: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        isBusy = true
        errorMessage = nil
        defer { isBusy = false }
        do {
            try await api.deleteSegmentPoint(caseId: caseId, side: side, branch: branch, index: currentPathIndex)
            let key = centerlineKey(side: side, branch: branch)
            acceptedIndices.remove(currentPathIndex)
            acceptedContoursMM[key]?[currentPathIndex] = nil
            acceptedContoursPx[key]?[currentPathIndex] = nil
            acceptedRadiiMM[key]?[currentPathIndex] = nil
            currentContourPx = []
            lastFittedRadiusMM = nil
        } catch {
            errorMessage = "Failed to delete segmentation: \(error.localizedDescription)"
        }
    }

    private func applySegmentResult(key: String, index: Int, result: APIClient.SegmentPointResult) {
        acceptedContoursMM[key, default: [:]][index] = result.contourMM
        acceptedContoursPx[key, default: [:]][index] = result.contourPx
        if let fitted = result.fittedRadiusMM {
            acceptedRadiiMM[key, default: [:]][index] = fitted
        }
    }

    func buildSurface(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        isBuildingSurface = true
        errorMessage = nil
        defer { isBuildingSurface = false }
        do {
            lastBuildResult = try await api.buildSurface(caseId: caseId, side: side)
        } catch {
            errorMessage = "Surface build failed: \(error.localizedDescription)"
            lastBuildResult = nil
        }
    }

    func saveSurfaceSTL(side: String, to url: URL) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            let data = try await api.fetchMeshSTL(caseId: caseId, side: side)
            try data.write(to: url)
        } catch {
            errorMessage = "Failed to save STL: \(error.localizedDescription)"
        }
    }

    /// su2MeshState/su2RunState/su2PostprocessState are each a single
    /// published property, not keyed per side — switching the selected
    /// side therefore used to go on showing whatever side was last acted
    /// on (CONFIRMED: a side that was never built or run could read "CFD
    /// run complete" right after switching to it, just because the
    /// *other* side's results were still sitting in these properties).
    /// Re-fetching all three for whichever side is now selected is the
    /// fix — every one of these already has its own per-side status
    /// endpoint, this just asks them on every switch instead of only
    /// right after starting a job. Best-effort per field: a transient
    /// fetch failure just leaves that one indicator at its previous
    /// (stale, but no worse than before this existed) value rather than
    /// failing the whole switch.
    func refreshSU2StatusForSide(_ side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        if let status = try? await api.fetchSU2MeshStatus(caseId: caseId, side: side) {
            su2MeshState = status.state
            su2MeshFaceMap = status.faceMap
            su2MeshError = status.error
            su2MeshStatusJSON = status.rawJSON
        }
        if let status = try? await api.fetchSU2RunStatus(caseId: caseId, side: side) {
            su2RunState = status.state
            su2RunStatus = status.state == "idle" ? nil : status
        }
        if let status = try? await api.fetchSU2PostprocessStatus(caseId: caseId, side: side) {
            su2PostprocessState = status.state
        }
    }

    /// Kicks off the CFD-ready volume mesh (.vtu) + boundary surface
    /// (.vtp, ModelFaceID-tagged) build and polls until it finishes —
    /// still a few seconds of real TetGen work, so this isn't a simple
    /// single request/response.
    func startSU2Mesh(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        su2MeshState = "running"
        su2MeshError = nil
        su2MeshFaceMap = nil
        su2MeshStatusJSON = nil
        // A rebuilt mesh describes different geometry — any previous run
        // or postprocessed result (including one just restored from a
        // saved project) is for the mesh being replaced, not this one.
        // Without resetting these here too, "View Results" can still
        // read su2PostprocessState == "done" from the old job and show
        // stale hemodynamics before the next status poll would catch up
        // (CONFIRMED: this is exactly how a real session ended up
        // looking at a previous mesh's own CFD results after tracing a
        // fresh path — see server/app.py's matching reset in
        // start_su2_mesh for the backend half of this fix).
        su2RunState = "idle"
        su2RunStatus = nil
        su2PostprocessState = "idle"
        do {
            try await api.startSU2Mesh(caseId: caseId, side: side)
        } catch {
            su2MeshState = "error"
            su2MeshError = error.localizedDescription
            return
        }
        while true {
            // A real TetGen run on a patient-size surface finishes in a
            // couple of seconds — polling every 2s (the old interval)
            // would often miss every intermediate stage entirely and
            // jump straight from "running" to "done". 300ms actually
            // catches the tagging/tetrahedralizing stages for the
            // debug panel to show.
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard let status = try? await api.fetchSU2MeshStatus(caseId: caseId, side: side) else { continue }
            su2MeshStatusJSON = status.rawJSON
            if status.state == "running" { continue }
            su2MeshState = status.state
            su2MeshFaceMap = status.faceMap
            su2MeshError = status.error
            break
        }
    }

    /// Starts the actual SU2 pulsatile CFD solve for this side (not just
    /// the mesh) and polls until it finishes, errors, or is cancelled.
    /// Polled at 1s (not su2 mesh's 300ms) — a real solve runs for
    /// minutes, not seconds, so there's no benefit to a tighter loop and
    /// it just means more idle HTTP round trips.
    func startSU2Run(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        su2RunState = "running"
        su2RunStatus = nil
        errorMessage = nil
        // This run's own results supersede whatever the previous run (or
        // a restored-from-disk one) left behind — see startSU2Mesh's
        // matching reset for why this can't just wait for the next
        // status poll.
        su2PostprocessState = "idle"
        do {
            try await api.startSU2Run(caseId: caseId, side: side)
        } catch {
            su2RunState = "error"
            errorMessage = "Failed to start CFD run: \(error.localizedDescription)"
            return
        }
        while true {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard let status = try? await api.fetchSU2RunStatus(caseId: caseId, side: side) else { continue }
            su2RunStatus = status
            su2RunState = status.state
            if status.state == "running" || status.state == "cancelling" { continue }
            break
        }
    }

    func cancelSU2Run(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            try await api.cancelSU2Run(caseId: caseId, side: side)
        } catch {
            errorMessage = "Failed to cancel CFD run: \(error.localizedDescription)"
        }
    }

    /// Kicks off (or re-kicks-off) extracting wall hemodynamics from
    /// this side's CFD run and polls until it's ready, then opens the
    /// results viewer — CFDResultsView does its own fetch of the actual
    /// payload once state reaches "done", this just owns the job
    /// lifecycle + sheet presentation.
    func startSU2Postprocess(side: String) async {
        guard let caseId = caseInfo?.caseId else { return }
        su2PostprocessState = "running"
        errorMessage = nil
        do {
            try await api.startSU2Postprocess(caseId: caseId, side: side)
        } catch {
            su2PostprocessState = "error"
            errorMessage = "Failed to start results extraction: \(error.localizedDescription)"
            return
        }
        while true {
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard let status = try? await api.fetchSU2PostprocessStatus(caseId: caseId, side: side) else { continue }
            su2PostprocessState = status.state
            if status.state == "running" { continue }
            if status.state == "error" {
                errorMessage = status.error ?? "Results extraction failed"
            } else if status.state == "done" {
                isCFDResultsPresented = true
            }
            break
        }
    }

    func saveSU2Mesh(side: String, vtuURL: URL, vtpURL: URL) async {
        guard let caseId = caseInfo?.caseId else { return }
        do {
            let vtu = try await api.fetchSU2MeshFile(caseId: caseId, side: side, kind: "vtu")
            let vtp = try await api.fetchSU2MeshFile(caseId: caseId, side: side, kind: "vtp")
            try vtu.write(to: vtuURL)
            try vtp.write(to: vtpURL)
        } catch {
            errorMessage = "Failed to save mesh: \(error.localizedDescription)"
        }
    }

    func resetWindowToDefault() {
        guard let info = caseInfo else { return }
        windowLevel = info.defaultWindow.wl
        windowWidth = info.defaultWindow.ww
    }

    // Centralized here (not recomputed ad hoc per view) so the Slider
    // bindings that use these can't observe a range that's out of sync with
    // what set the current value — SwiftUI's Slider crashes with a
    // precondition failure if its bound value is ever outside `in:` at
    // render time (confirmed via a real crash report: Normalizing.init
    // precondition, triggered from a stale/mismatched window-width range).
    var windowLevelRange: ClosedRange<Double> {
        guard let info = caseInfo else { return 0...1 }
        let span = max(info.defaultWindow.ww, 100)
        return (info.defaultWindow.wl - span)...(info.defaultWindow.wl + span)
    }

    var windowWidthRange: ClosedRange<Double> {
        guard let info = caseInfo else { return 1...2 }
        return 1...max(info.defaultWindow.ww * 3, 100)
    }
}
