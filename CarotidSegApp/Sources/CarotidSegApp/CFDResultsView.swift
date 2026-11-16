import SceneKit
import SwiftUI

/// Same 5-stop WSS colormap the validated standalone pipeline's own
/// desktop viewer used (~/Desktop/SU2 Pipeline Code/agibalova_right/
/// wss_viz/app.py: dark blue -> teal -> green -> amber -> red), reused
/// here for every scalar field (not just WSS) so results from this app
/// read consistently with that established convention rather than
/// introducing a second, different palette.
enum CFDColormap {
    private static func rgb(_ r: Int, _ g: Int, _ b: Int) -> SCNVector3 {
        SCNVector3(Float(r) / 255, Float(g) / 255, Float(b) / 255)
    }

    private static let stops: [(Double, SCNVector3)] = [
        (0.00, rgb(0x24, 0x30, 0x6B)),
        (0.25, rgb(0x1F, 0x73, 0xA8)),
        (0.50, rgb(0x21, 0xA1, 0x8C)),
        (0.75, rgb(0xE8, 0xA9, 0x3D)),
        (1.00, rgb(0xC4, 0x40, 0x2F)),
    ]

    /// t in [0,1] -> RGB in [0,1]. Values outside [0,1] clamp to the
    /// nearest end color (a threshold-filtered value is given its own
    /// dimmed gray instead of calling this — see dimmedGray).
    static func color(_ t: Double) -> SCNVector3 {
        let tc: CGFloat = CGFloat(min(max(t, 0), 1))
        for i in 1..<stops.count {
            if tc <= CGFloat(stops[i].0) {
                let (t0d, c0) = stops[i - 1]
                let (t1d, c1) = stops[i]
                let t0 = CGFloat(t0d), t1 = CGFloat(t1d)
                let f: CGFloat = t1 > t0 ? (tc - t0) / (t1 - t0) : 0
                let x: CGFloat = c0.x + (c1.x - c0.x) * f
                let y: CGFloat = c0.y + (c1.y - c0.y) * f
                let z: CGFloat = c0.z + (c1.z - c0.z) * f
                return SCNVector3(x, y, z)
            }
        }
        return stops.last!.1
    }

    /// What a point outside the active threshold band renders as — dim
    /// and desaturated so the highlighted band reads as clearly singled
    /// out, without hiding the rest of the vessel's own shape entirely.
    static let dimmedGray = SCNVector3(0.22, 0.24, 0.27)
}

/// The six hemodynamic fields this viewer can color/threshold by — the
/// derived ones (TAWSS/OSI/RRT/mean pressure) come straight off the main
/// /su2_postprocess/result payload; the instantaneous ones are fetched
/// one frame at a time from /su2_postprocess/frame (see that endpoint's
/// own docstring for why — a full per-frame transfer for a real run's
/// frame count is tens of MB).
enum CFDField: String, CaseIterable, Identifiable {
    case tawss, osi, rrt, meanPressure, instantWSS, instantPressure
    var id: String { rawValue }

    var label: String {
        switch self {
        case .tawss: return "TAWSS"
        case .osi: return "OSI"
        case .rrt: return "RRT"
        case .meanPressure: return "Mean Pressure"
        case .instantWSS: return "Instantaneous WSS"
        case .instantPressure: return "Instantaneous Pressure"
        }
    }

    var unit: String {
        switch self {
        case .tawss, .instantWSS: return "Pa"
        case .osi: return ""
        case .rrt: return "1/Pa"
        case .meanPressure, .instantPressure: return "Pa"
        }
    }

    var isInstantaneous: Bool { self == .instantWSS || self == .instantPressure }
}

@MainActor
final class CFDResultsLoader: ObservableObject {
    @Published var result: APIClient.SU2PostprocessResult?
    @Published var isLoading = false
    @Published var loadError: String?
    @Published var frameValues: [Double]?
    @Published var isLoadingFrame = false

    private(set) var wallMesh: CFDWallMesh?
    private(set) var frameRevision = 0

    private let api: APIClient
    private let caseId: String
    private let side: String

    init(api: APIClient, caseId: String, side: String) {
        self.api = api
        self.caseId = caseId
        self.side = side
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let loaded = try await api.fetchSU2PostprocessResult(caseId: caseId, side: side)
            wallMesh = CFDWallMesh(pointsMM: loaded.pointsMM, triangles: loaded.triangles)
            result = loaded
        } catch {
            loadError = "Failed to load results: \(error.localizedDescription)"
        }
    }

    func loadFrame(index: Int, field: CFDField) async {
        isLoadingFrame = true
        defer { isLoadingFrame = false }
        let apiField = field == .instantWSS ? "wss" : "pressure"
        do {
            let values = try await api.fetchSU2PostprocessFrame(caseId: caseId, side: side, index: index, field: apiField)
            frameRevision += 1
            frameValues = values
        } catch {
            loadError = "Failed to load frame: \(error.localizedDescription)"
        }
    }
}

/// CFD results viewer: the wall mesh colored by a chosen hemodynamic
/// field, with a threshold band that highlights the matching region and
/// reports its own area live — plus the geometry "sizes" panel (wall
/// area, cap areas/diameters). Reuses SurfaceGeometryView (built for the
/// STL preview) for the actual 3D render since it already handles the
/// camera/orbit-controls plumbing; only the geometry it's given is new.
struct CFDResultsView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    let side: String

    @StateObject private var loader: CFDResultsLoader
    @StateObject private var controller = Scene3DController()
    @State private var field: CFDField = .tawss
    @State private var frameIndex: Int = 0
    @State private var thresholdEnabled = false
    @State private var thresholdLow: Double = 0
    @State private var thresholdHigh: Double = 1
    @State private var isSavingResults = false
    @State private var didSaveResults = false

    init(side: String, api: APIClient, caseId: String) {
        self.side = side
        _loader = StateObject(wrappedValue: CFDResultsLoader(api: api, caseId: caseId, side: side))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            HStack(alignment: .top, spacing: 14) {
                viewerPane
                    .frame(minWidth: 560, minHeight: 620)
                sidePanel
                    .frame(width: 320)
            }
            if let error = loader.loadError {
                Text(error).foregroundStyle(.red).font(Theme.labelFont(11))
            }
        }
        .padding(20)
        .frame(minWidth: 940, minHeight: 700)
        .background(Theme.background)
        .task {
            await loader.load()
            resetThresholdToFullRange()
        }
        .onChange(of: field) { _ in
            resetThresholdToFullRange()
            if field.isInstantaneous {
                Task { await loader.loadFrame(index: frameIndex, field: field) }
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("CFD Results — \(side.capitalized)")
                    .font(Theme.labelFont(18, weight: .bold))
                    .foregroundStyle(Theme.ink)
                Text("Wall hemodynamics from the finished CFD run — color by field, threshold a band, read its area.")
                    .font(Theme.labelFont(11))
                    .foregroundStyle(Theme.inkMuted)
            }
            Spacer()
            Button {
                Task {
                    isSavingResults = true
                    await state.saveCFDResultsToProject(side: side)
                    isSavingResults = false
                    didSaveResults = true
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    didSaveResults = false
                }
            } label: {
                if isSavingResults {
                    ProgressView().controlSize(.small)
                } else if didSaveResults {
                    Label("Saved", systemImage: "checkmark")
                } else {
                    Text("Save Results…")
                }
            }
            .buttonStyle(.themed)
            .disabled(isSavingResults)

            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.themed)
        }
    }

    private var viewerPane: some View {
        ZStack {
            Color.black
            if let geometry = coloredGeometry {
                SurfaceGeometryView(geometry: geometry, controller: controller)
                    .overlay(alignment: .topTrailing) {
                        Scene3DControlsOverlay(controller: controller)
                    }
                    .overlay(alignment: .bottomLeading) {
                        colorLegend
                    }
            } else if loader.isLoading {
                ProgressView("Loading results…").foregroundStyle(.white)
            } else if loader.loadError == nil {
                Text("No results yet.").foregroundStyle(.white.opacity(0.6))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    /// Current field's own value array per wall point, whatever its
    /// source (a static derived array already in `result`, or the one
    /// frame fetched on demand for an instantaneous field).
    private var activeValues: [Double]? {
        guard let result = loader.result else { return nil }
        switch field {
        case .tawss: return result.tawssPa
        case .osi: return result.osi
        case .rrt: return result.rrtPaInv
        case .meanPressure: return result.meanPressurePa
        case .instantWSS, .instantPressure: return loader.frameValues
        }
    }

    private var valueRevision: Int { field.isInstantaneous ? loader.frameRevision : 0 }

    private var valueRange: ClosedRange<Double> {
        guard let mesh = loader.wallMesh, let values = activeValues else { return 0...1 }
        return mesh.valueRange(values: values, field: field, revision: valueRevision)
    }

    private var coloredGeometry: SCNGeometry? {
        guard let mesh = loader.wallMesh, let values = activeValues else { return nil }
        let band: ClosedRange<Double>? = thresholdEnabled ? thresholdLow...max(thresholdHigh, thresholdLow) : nil
        return mesh.geometry(
            values: values, field: field, revision: valueRevision,
            range: valueRange, threshold: band
        )
    }

    private var colorLegend: some View {
        HStack(spacing: 6) {
            Text(String(format: "%.2f", valueRange.lowerBound))
            LinearGradient(
                colors: stride(from: 0.0, through: 1.0, by: 0.1).map {
                    let c = CFDColormap.color($0)
                    return Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
                },
                startPoint: .leading, endPoint: .trailing
            )
            .frame(width: 140, height: 10)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            Text(String(format: "%.2f", valueRange.upperBound))
            Text(field.unit).foregroundStyle(.white.opacity(0.6))
        }
        .font(Theme.readoutFont(10))
        .foregroundStyle(.white)
        .padding(8)
    }

    private var sidePanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                fieldSection
                Divider().background(Theme.ink.opacity(0.15))
                if field.isInstantaneous {
                    frameSection
                    Divider().background(Theme.ink.opacity(0.15))
                }
                thresholdSection
                Divider().background(Theme.ink.opacity(0.15))
                geometrySection
            }
            .padding(.trailing, 4)
        }
    }

    private var fieldSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("FIELD").font(Theme.labelFont(11, weight: .bold)).foregroundStyle(Theme.inkMuted)
            Picker("", selection: $field) {
                ForEach(CFDField.allCases) { f in
                    Text(f.label).tag(f)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
        }
    }

    private var frameSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("TIME").font(Theme.labelFont(11, weight: .bold)).foregroundStyle(Theme.inkMuted)
            if let result = loader.result, !result.timesS.isEmpty {
                Slider(
                    value: Binding(
                        get: { Double(frameIndex) },
                        set: { frameIndex = Int($0.rounded()) }
                    ),
                    in: 0...Double(result.timesS.count - 1),
                    step: 1
                ) { isEditing in
                    if !isEditing { Task { await loader.loadFrame(index: frameIndex, field: field) } }
                }
                HStack {
                    if loader.isLoadingFrame { ProgressView().controlSize(.small) }
                    Text(String(format: "t = %.3fs  (frame %d/%d)", result.timesS[frameIndex], frameIndex + 1, result.timesS.count))
                        .font(Theme.readoutFont(11))
                        .foregroundStyle(Theme.ink)
                }
            }
        }
    }

    private var thresholdSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("THRESHOLD FILTER").font(Theme.labelFont(11, weight: .bold)).foregroundStyle(Theme.inkMuted)
                Spacer()
                Toggle("", isOn: $thresholdEnabled).labelsHidden().toggleStyle(.switch).controlSize(.small)
            }
            if thresholdEnabled {
                paramSlider(label: "Min", value: $thresholdLow, range: valueRange)
                paramSlider(label: "Max", value: $thresholdHigh, range: valueRange)
                if field == .tawss || field == .instantWSS {
                    HStack(spacing: 8) {
                        Button("Low WSS (<0.4 Pa)") { thresholdLow = valueRange.lowerBound; thresholdHigh = 0.4 }
                            .buttonStyle(.themed)
                        Button("High WSS (>5 Pa)") { thresholdLow = 5.0; thresholdHigh = valueRange.upperBound }
                            .buttonStyle(.themed)
                    }
                    .font(Theme.labelFont(10))
                }
                thresholdAreaReadout
            }
        }
    }

    private var thresholdAreaReadout: some View {
        let (areaMM2, fraction) = thresholdAreaStats()
        return VStack(alignment: .leading, spacing: 2) {
            ReadoutLabel(
                value: String(format: "%.1f mm²", areaMM2),
                caption: String(format: "%.1f%% of wall area", fraction * 100),
                size: 13
            )
        }
    }

    /// Sums the area of every triangle whose *every* vertex falls in the
    /// active threshold band — a triangle straddling the boundary is
    /// excluded rather than counted whole, so this stays a lower (never
    /// an over-) estimate of the highlighted region's real area.
    private func thresholdAreaStats() -> (area: Double, fraction: Double) {
        guard let mesh = loader.wallMesh, let values = activeValues else { return (0, 0) }
        let band = min(thresholdLow, thresholdHigh)...max(thresholdLow, thresholdHigh)
        return mesh.thresholdArea(values: values, range: band)
    }

    private func paramSlider(label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack {
            Text(label).font(Theme.labelFont(11)).foregroundStyle(Theme.ink).frame(width: 36, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.2f", value.wrappedValue))
                .font(Theme.readoutFont(11))
                .foregroundStyle(Theme.textBright)
                .frame(width: 60, alignment: .trailing)
        }
    }

    private var geometrySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("GEOMETRY").font(Theme.labelFont(11, weight: .bold)).foregroundStyle(Theme.inkMuted)
            if let geom = loader.result?.geometry {
                ReadoutLabel(value: String(format: "%.1f mm²", geom.wallAreaMm2), caption: "wall area", size: 13)
                ForEach(geom.caps.sorted(by: { $0.key < $1.key }), id: \.key) { name, cap in
                    ReadoutLabel(
                        value: String(format: "⌀ %.2f mm", cap.equivalentDiameterMm),
                        caption: "\(name) — \(String(format: "%.1f mm²", cap.areaMm2))",
                        size: 13
                    )
                }
            }
        }
    }

    private func resetThresholdToFullRange() {
        let r = valueRange
        thresholdLow = r.lowerBound
        thresholdHigh = r.upperBound
    }
}
