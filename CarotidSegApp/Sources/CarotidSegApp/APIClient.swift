import AppKit
import Foundation

struct SeriesInfo: Decodable, Identifiable {
    let seriesUID: String
    let description: String
    let modality: String
    let nFiles: Int
    var id: String { seriesUID }

    enum CodingKeys: String, CodingKey {
        case seriesUID = "series_uid"
        case description
        case modality
        case nFiles = "n_files"
    }
}

struct DefaultWindow: Decodable {
    let wl: Double
    let ww: Double
}

struct CaseInfo: Decodable {
    let caseId: String
    let shape: [Int]
    let spacingMM: [Double]
    let originMM: [Double]
    let modality: String
    let defaultWindow: DefaultWindow

    enum CodingKeys: String, CodingKey {
        case caseId = "case_id"
        case shape
        case spacingMM = "spacing_mm"
        case originMM = "origin_mm"
        case modality
        case defaultWindow = "default_window"
    }
}

enum APIError: LocalizedError {
    case server(String)
    case badResponse
    case badImage

    var errorDescription: String? {
        switch self {
        case .server(let detail): return detail
        case .badResponse: return "Unexpected response from backend"
        case .badImage: return "Backend returned an invalid image"
        }
    }
}

@MainActor
/// Stateless aside from an immutable baseURL — safe to call concurrently
/// (seedCircularContours in AppState fires many segmentPoint calls at
/// once via a TaskGroup).
final class APIClient: @unchecked Sendable {
    let baseURL = URL(string: "http://127.0.0.1:8000")!

    private func decodeErrorDetail(_ data: Data) -> String? {
        struct Err: Decodable { let detail: String }
        return (try? JSONDecoder().decode(Err.self, from: data))?.detail
    }

    func listSeries(path: String) async throws -> [SeriesInfo] {
        var components = URLComponents(url: baseURL.appendingPathComponent("series"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "path", value: path)]
        let (data, response) = try await URLSession.shared.data(from: components.url!)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let series: [SeriesInfo] }
        return try JSONDecoder().decode(Wrapper.self, from: data).series
    }

    func createCase(path: String, seriesUID: String?) async throws -> CaseInfo {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["path": path]
        if let seriesUID { body["series_uid"] = seriesUID }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(CaseInfo.self, from: data)
    }

    struct SideRestoreInfo: Decodable {
        let centerline: [String]
        let segmentation: [String]
        let surfaceFileExists: Bool
        let meshFilesExist: Bool
        // true when a saved CFD result (server/app.py's
        // save_cfd_results_to_project) was found and restored directly
        // into the backend's su2_postprocess_jobs — see AppState.openProject.
        let cfdResults: Bool

        enum CodingKeys: String, CodingKey {
            case centerline, segmentation
            case surfaceFileExists = "surface_file_exists"
            case meshFilesExist = "mesh_files_exist"
            case cfdResults = "cfd_results"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            centerline = try c.decode([String].self, forKey: .centerline)
            segmentation = try c.decode([String].self, forKey: .segmentation)
            surfaceFileExists = try c.decode(Bool.self, forKey: .surfaceFileExists)
            meshFilesExist = try c.decode(Bool.self, forKey: .meshFilesExist)
            cfdResults = try c.decodeIfPresent(Bool.self, forKey: .cfdResults) ?? false
        }
    }

    struct OpenProjectResult {
        let caseInfo: CaseInfo
        let projectDir: String?
        let restored: [String: SideRestoreInfo]
    }

    /// Reopens a project folder saved by the per-step Save buttons (see
    /// AppState.save*ToProject) — reloads the same DICOM series fresh and
    /// restores whatever centerline/segmentation files that folder has.
    /// See server/app.py's /projects/open for why a saved surface/mesh
    /// isn't restored the same way (both just get rebuilt on demand).
    func openProject(path: String) async throws -> OpenProjectResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("projects/open"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["path": path])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let caseInfo = try JSONDecoder().decode(CaseInfo.self, from: data)
        struct Wrapper: Decodable { let project_dir: String?; let restored: [String: SideRestoreInfo] }
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        return OpenProjectResult(caseInfo: caseInfo, projectDir: w.project_dir, restored: w.restored)
    }

    struct SeriesPointResult: Decodable {
        let contour_px: [[Double]]?
        let contour_mm: [[Double]]?
        let radius_mm: Double?
    }

    /// The full already-segmented Series for one branch — used to
    /// repopulate AppState's acceptedContours*/acceptedRadiiMM after
    /// AppState.openProject restores a case's segmentation_series
    /// server-side, since that restore doesn't itself push anything into
    /// the client's own per-point dictionaries.
    func fetchSegmentationSeries(caseId: String, side: String, branch: String) async throws -> [SeriesPointResult] {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let points: [SeriesPointResult] }
        return try JSONDecoder().decode(Wrapper.self, from: data).points
    }

    func fetchSlice(caseId: String, axis: PlaneAxis, index: Int, wl: Double, ww: Double) async throws -> NSImage {
        var components = URLComponents(
            url: baseURL.appendingPathComponent("cases/\(caseId)/slice"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [
            URLQueryItem(name: "axis", value: axis.rawValue),
            URLQueryItem(name: "index", value: String(index)),
            URLQueryItem(name: "wl", value: String(wl)),
            URLQueryItem(name: "ww", value: String(ww)),
        ]
        let (data, response) = try await URLSession.shared.data(from: components.url!)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        guard let image = NSImage(data: data) else { throw APIError.badImage }
        return image
    }

    struct CenterlineResult {
        let path: [[Int]]
        let controlPoints: [[Int]]
    }

    private struct CenterlineWrapper: Decodable {
        let path: [[Int]]
        let control_points: [[Int]]
    }

    func addCenterlinePoint(caseId: String, side: String, branch: String, point: [Int]) async throws -> CenterlineResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)/points"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["point": point])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let w = try JSONDecoder().decode(CenterlineWrapper.self, from: data)
        return CenterlineResult(path: w.path, controlPoints: w.control_points)
    }

    /// Full point-to-point auto-trace between exactly two clicked
    /// endpoints — replaces whatever path this branch already had. See
    /// server/app.py's /auto_trace for the solver and its known
    /// tortuosity caveat.
    func autoTraceCenterline(caseId: String, side: String, branch: String, start: [Int], end: [Int]) async throws -> CenterlineResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)/auto_trace"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["start": start, "end": end])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let w = try JSONDecoder().decode(CenterlineWrapper.self, from: data)
        return CenterlineResult(path: w.path, controlPoints: w.control_points)
    }

    func moveCenterlinePoint(caseId: String, side: String, branch: String, index: Int, point: [Int]) async throws -> CenterlineResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)/points/\(index)"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["point": point])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let w = try JSONDecoder().decode(CenterlineWrapper.self, from: data)
        return CenterlineResult(path: w.path, controlPoints: w.control_points)
    }

    func deleteCenterlinePoint(caseId: String, side: String, branch: String, index: Int) async throws -> CenterlineResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)/points/\(index)"))
        request.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let w = try JSONDecoder().decode(CenterlineWrapper.self, from: data)
        return CenterlineResult(path: w.path, controlPoints: w.control_points)
    }

    func reorderCenterlinePoint(caseId: String, side: String, branch: String, fromIndex: Int, toIndex: Int) async throws -> CenterlineResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)/reorder"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["from_index": fromIndex, "to_index": toIndex])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        let w = try JSONDecoder().decode(CenterlineWrapper.self, from: data)
        return CenterlineResult(path: w.path, controlPoints: w.control_points)
    }

    func fetchControlPoints(caseId: String) async throws -> [String: [[Int]]] {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("cases/\(caseId)/control_points"))
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode([String: [[Int]]].self, from: data)
    }

    func clearCenterline(caseId: String, side: String, branch: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/centerline/\(side)/\(branch)"))
        request.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    func fetchCenterlines(caseId: String) async throws -> [String: [[Int]]] {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("cases/\(caseId)/centerline"))
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode([String: [[Int]]].self, from: data)
    }

    struct PrepareSeriesResult {
        let nPoints: Int
        let pointsMM: [[Double]]
        let tangentsMM: [[Double]]
    }

    func prepareSeries(caseId: String, side: String, branch: String) async throws -> PrepareSeriesResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)/prepare"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let n_points: Int; let points_mm: [[Double]]; let tangents_mm: [[Double]] }
        let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
        return PrepareSeriesResult(nPoints: wrapper.n_points, pointsMM: wrapper.points_mm, tangentsMM: wrapper.tangents_mm)
    }

    func fetchSegmentSlice(caseId: String, side: String, branch: String, index: Int, wl: Double, ww: Double) async throws -> NSImage {
        var components = URLComponents(
            url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)/slice/\(index)"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [
            URLQueryItem(name: "wl", value: String(wl)),
            URLQueryItem(name: "ww", value: String(ww)),
        ]
        let (data, response) = try await URLSession.shared.data(from: components.url!)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        guard let image = NSImage(data: data) else { throw APIError.badImage }
        return image
    }

    struct SegmentPointResult {
        let contourPx: [[Double]]
        let contourMM: [[Double]]
        let fittedRadiusMM: Double?
    }

    func segmentPoint(caseId: String, side: String, branch: String, index: Int,
                       method: String, radiusMM: Double) async throws -> SegmentPointResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)/point/\(index)"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["method": method, "radius_mm": radiusMM])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let contour_px: [[Double]]; let contour_mm: [[Double]]; let fitted_radius_mm: Double? }
        let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
        return SegmentPointResult(contourPx: wrapper.contour_px, contourMM: wrapper.contour_mm, fittedRadiusMM: wrapper.fitted_radius_mm)
    }

    func segmentPointManual(caseId: String, side: String, branch: String, index: Int,
                             contourPx: [[Double]]) async throws -> SegmentPointResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)/point/\(index)/manual"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["contour_px": contourPx])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let contour_px: [[Double]]; let contour_mm: [[Double]]; let fitted_radius_mm: Double? }
        let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
        return SegmentPointResult(contourPx: wrapper.contour_px, contourMM: wrapper.contour_mm, fittedRadiusMM: wrapper.fitted_radius_mm)
    }

    func deleteSegmentPoint(caseId: String, side: String, branch: String, index: Int) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/\(branch)/point/\(index)"))
        request.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    struct BuildSurfaceResult {
        let ok: Bool
        let nPoints: Int?
        let nCells: Int?
        let nBoundaryLoops: Int?
        let cleanTopology: Bool?
        let error: String?
        // Every field the backend's diag dict returned, pretty-printed —
        // branches_used, bbox_margin_mm, per-branch points_stamped/
        // substituted_outlier, n_nonmanifold_edges, hole_fill_size_mm, etc.
        // The named fields above cover the common case; this is for the
        // debug panel, which shows the whole thing rather than guessing
        // in advance which diag field will matter for a given build.
        let rawJSON: String
    }

    func buildSurface(caseId: String, side: String) async throws -> BuildSurfaceResult {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/build_surface"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        if http.statusCode != 200 {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable {
            let ok: Bool?
            let n_points: Int?
            let n_cells: Int?
            let n_boundary_loops: Int?
            let clean_topology: Bool?
            let error: String?
        }
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        let rawJSON = prettyPrintedJSON(data)
        return BuildSurfaceResult(
            ok: w.ok ?? false, nPoints: w.n_points, nCells: w.n_cells,
            nBoundaryLoops: w.n_boundary_loops, cleanTopology: w.clean_topology, error: w.error,
            rawJSON: rawJSON
        )
    }

    private func prettyPrintedJSON(_ data: Data) -> String {
        guard let obj = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: pretty, encoding: .utf8) else {
            return String(data: data, encoding: .utf8) ?? ""
        }
        return text
    }

    func fetchMeshSTL(caseId: String, side: String) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("cases/\(caseId)/mesh/\(side)"))
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return data
    }

    func startSU2Mesh(caseId: String, side: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_mesh"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    struct SU2MeshStatus {
        let state: String
        let faceMap: [String: String]?
        let error: String?
        // Every field the backend's diag/stage_info dicts actually
        // return varies by stage (tagging vs. tetrahedralizing vs. done)
        // and isn't worth a fixed Decodable shape — same reasoning as
        // BuildSurfaceResult.rawJSON — so the debug panel just displays
        // this directly rather than needing a field added here (and in
        // the backend's response model) every time a new stat is added.
        let rawJSON: String
    }

    func fetchSU2MeshStatus(caseId: String, side: String) async throws -> SU2MeshStatus {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_mesh/status")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let state: String; let face_map: [String: String]?; let error: String? }
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        return SU2MeshStatus(state: w.state, faceMap: w.face_map, error: w.error, rawJSON: prettyPrintedJSON(data))
    }

    func fetchSU2MeshFile(caseId: String, side: String, kind: String) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_mesh/\(kind)")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return data
    }

    /// The finished mesh's own boundary surface, converted server-side to
    /// binary STL purely for SurfacePreviewView to show — same endpoint
    /// family as fetchSU2MeshFile(kind:), just named for what it's for.
    func fetchSU2MeshPreviewSTL(caseId: String, side: String) async throws -> Data {
        try await fetchSU2MeshFile(caseId: caseId, side: side, kind: "preview_stl")
    }

    /// Cap diameters/centroids/normals + inter-cap angles for the mesh
    /// preview's own callouts — see server/app.py's su2_mesh/geometry.
    func fetchSU2MeshGeometry(caseId: String, side: String) async throws -> SU2MeshGeometry {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_mesh/geometry")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(SU2MeshGeometry.self, from: data)
    }

    // MARK: - SU2 CFD run (the actual solve, not just the mesh)

    struct SU2RunStatus {
        let state: String
        let step: Int?
        let totalSteps: Int?
        let tSim: Double?
        let cycleS: Double?
        let elapsedS: Double?
        let qIcaMlps: Double?
        let qEcaMlps: Double?
        let pIcaMmHg: Double?
        let pEcaMmHg: Double?
        let error: String?
        let outputDir: String?
        let mpiProcesses: Int?

        /// Backend reports raw elapsed time + step count, not a
        /// precomputed ETA — a simple linear extrapolation (time per
        /// step so far * steps remaining) computed here is just as
        /// accurate and saves a field the backend would otherwise need
        /// to keep in sync.
        var etaSeconds: Double? {
            guard let step, step > 0, let totalSteps, let elapsedS else { return nil }
            let remaining = totalSteps - step
            guard remaining > 0 else { return 0 }
            return elapsedS / Double(step) * Double(remaining)
        }

        var fractionComplete: Double {
            guard let step, let totalSteps, totalSteps > 0 else { return 0 }
            return min(1.0, Double(step) / Double(totalSteps))
        }
    }

    @discardableResult
    func startSU2Run(caseId: String, side: String) async throws -> Int {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_run"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let total_steps: Int }
        return try JSONDecoder().decode(Wrapper.self, from: data).total_steps
    }

    func fetchSU2RunStatus(caseId: String, side: String) async throws -> SU2RunStatus {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_run/status")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable {
            let state: String
            let step: Int?
            let total_steps: Int?
            let t_sim: Double?
            let cycle_s: Double?
            let elapsed_s: Double?
            let q_ica_mlps: Double?
            let q_eca_mlps: Double?
            let p_ica_mmhg: Double?
            let p_eca_mmhg: Double?
            let error: String?
            let output_dir: String?
            let mpi_processes: Int?
        }
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        return SU2RunStatus(
            state: w.state, step: w.step, totalSteps: w.total_steps, tSim: w.t_sim, cycleS: w.cycle_s,
            elapsedS: w.elapsed_s, qIcaMlps: w.q_ica_mlps, qEcaMlps: w.q_eca_mlps,
            pIcaMmHg: w.p_ica_mmhg, pEcaMmHg: w.p_eca_mmhg, error: w.error,
            outputDir: w.output_dir, mpiProcesses: w.mpi_processes
        )
    }

    func cancelSU2Run(caseId: String, side: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_run/cancel"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    // MARK: - CFD results (post-processed wall hemodynamics)

    struct SU2PostprocessStatus {
        let state: String
        let frame: Int?
        let totalFrames: Int?
        let error: String?
    }

    struct CapStats: Decodable {
        let areaMm2: Double
        let equivalentDiameterMm: Double
        let centroidMm: [Double]
        // Only present on the mesh-geometry endpoint (su2_mesh/geometry),
        // not on the CFD-results one — optional so the same type decodes
        // both without complaint.
        let normalMm: [Double]?

        enum CodingKeys: String, CodingKey {
            case areaMm2 = "area_mm2"
            case equivalentDiameterMm = "equivalent_diameter_mm"
            case centroidMm = "centroid_mm"
            case normalMm = "normal_mm"
        }
    }

    struct MeshAngle: Decodable {
        let label: String
        let degrees: Double
        let atMm: [Double]

        enum CodingKeys: String, CodingKey {
            case label, degrees
            case atMm = "at_mm"
        }
    }

    struct SU2PostprocessGeometry: Decodable {
        let wallAreaMm2: Double
        let caps: [String: CapStats]

        enum CodingKeys: String, CodingKey {
            case wallAreaMm2 = "wall_area_mm2"
            case caps
        }
    }

    /// Cap diameters/centroids/normals + inter-cap angles straight from a
    /// finished mesh (see server/app.py's su2_mesh/geometry) — fetched
    /// for the mesh preview's own diameter/angle callouts, no CFD run
    /// required (unlike SU2PostprocessGeometry, which only exists once a
    /// solve has run).
    struct SU2MeshGeometry: Decodable {
        let wallAreaMm2: Double
        let caps: [String: CapStats]
        let angles: [MeshAngle]

        enum CodingKeys: String, CodingKey {
            case wallAreaMm2 = "wall_area_mm2"
            case caps, angles
        }
    }

    struct SU2PostprocessResult: Decodable {
        let pointsMM: [[Double]]
        let triangles: [[Int]]
        let timesS: [Double]
        let settledCycleStartS: Double
        let tawssPa: [Double]
        let osi: [Double]
        let rrtPaInv: [Double]
        let meanPressurePa: [Double]
        let geometry: SU2PostprocessGeometry

        enum CodingKeys: String, CodingKey {
            case pointsMM = "points_mm"
            case triangles
            case timesS = "times_s"
            case settledCycleStartS = "settled_cycle_start_s"
            case tawssPa = "tawss_pa"
            case osi
            case rrtPaInv = "rrt_pa_inv"
            case meanPressurePa = "mean_pressure_pa"
            case geometry
        }
    }

    func startSU2Postprocess(caseId: String, side: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_postprocess"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    func fetchSU2PostprocessStatus(caseId: String, side: String) async throws -> SU2PostprocessStatus {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_postprocess/status")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let state: String; let frame: Int?; let total_frames: Int?; let error: String? }
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        return SU2PostprocessStatus(state: w.state, frame: w.frame, totalFrames: w.total_frames, error: w.error)
    }

    func fetchSU2PostprocessResult(caseId: String, side: String) async throws -> SU2PostprocessResult {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_postprocess/result")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(SU2PostprocessResult.self, from: data)
    }

    /// One instantaneous frame's per-wall-point values — field: "wss" or
    /// "pressure". See server/app.py's su2_postprocess_frame endpoint
    /// docstring for why this is fetched one frame at a time rather than
    /// bundled into /result.
    func fetchSU2PostprocessFrame(caseId: String, side: String, index: Int, field: String) async throws -> [Double] {
        var components = URLComponents(
            url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_postprocess/frame/\(index)"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "field", value: field)]
        let (data, response) = try await URLSession.shared.data(from: components.url!)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let index: Int; let t: Double; let values: [Double] }
        return try JSONDecoder().decode(Wrapper.self, from: data).values
    }

    // MARK: - Project folder + per-step Save

    @discardableResult
    func setProjectDir(caseId: String, path: String) async throws -> String? {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/project"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["path": path])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let project_dir: String? }
        return try JSONDecoder().decode(Wrapper.self, from: data).project_dir
    }

    func fetchProjectDir(caseId: String) async throws -> String? {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("cases/\(caseId)/project"))
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        struct Wrapper: Decodable { let project_dir: String? }
        return try JSONDecoder().decode(Wrapper.self, from: data).project_dir
    }

    /// Shared by every "Save" endpoint next to a workflow step — all take
    /// no body and return {"saved_to": ...}, which the caller doesn't
    /// currently need beyond confirming success (no 4xx/5xx).
    private func postSave(caseId: String, path: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/\(path)"))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
    }

    func saveCenterlineToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/centerline/\(side)")
    }

    func saveSegmentationToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/segmentation/\(side)")
    }

    func saveSurfaceToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/surface/\(side)")
    }

    func saveMeshToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/mesh/\(side)")
    }

    func saveSU2ParamsToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/su2_params/\(side)")
    }

    func saveCFDResultsToProject(caseId: String, side: String) async throws {
        try await postSave(caseId: caseId, path: "save/cfd_results/\(side)")
    }

    // MARK: - Boundary conditions (SU2 physiological parameters)

    struct SU2Params: Codable, Equatable {
        var psvCms: Double
        var edvCms: Double
        var cardiacCycleS: Double
        var icaFraction: Double
        // Independently editable per outlet — the two outlets' own
        // downstream vascular beds don't necessarily share one operating
        // pressure, unlike the single shared MAP this replaced.
        var icaMapMmHg: Double
        var ecaMapMmHg: Double

        enum CodingKeys: String, CodingKey {
            case psvCms = "psv_cms"
            case edvCms = "edv_cms"
            case cardiacCycleS = "cardiac_cycle_s"
            case icaFraction = "ica_fraction"
            case icaMapMmHg = "ica_map_mmhg"
            case ecaMapMmHg = "eca_map_mmhg"
        }
    }

    func fetchSU2Params(caseId: String, side: String) async throws -> SU2Params {
        let (data, response) = try await URLSession.shared.data(
            from: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_params")
        )
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(SU2Params.self, from: data)
    }

    @discardableResult
    func postSU2Params(caseId: String, side: String, params: SU2Params) async throws -> SU2Params {
        var request = URLRequest(url: baseURL.appendingPathComponent("cases/\(caseId)/segment/\(side)/su2_params"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(params)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard http.statusCode == 200 else {
            throw APIError.server(decodeErrorDetail(data) ?? "HTTP \(http.statusCode)")
        }
        return try JSONDecoder().decode(SU2Params.self, from: data)
    }
}
