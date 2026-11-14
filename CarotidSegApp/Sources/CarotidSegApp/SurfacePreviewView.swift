import SceneKit
import SwiftUI

/// Parses a binary STL (the only kind build_surface.py's write_stl
/// produces — SetFileTypeToBinary()) into flat triangle geometry: 80-byte
/// header, 4-byte triangle count, then 50 bytes per triangle (12 for a
/// normal we don't use — SceneKit recomputes its own — 36 for the 3
/// vertices, 2 trailing attribute bytes). Returns nil on anything that
/// doesn't look like a well-formed binary STL rather than crash on it.
func parseBinarySTL(_ data: Data) -> SCNGeometry? {
    let headerSize = 80
    let countSize = 4
    guard data.count >= headerSize + countSize else { return nil }
    let triCount = data.withUnsafeBytes { raw -> UInt32 in
        raw.loadUnaligned(fromByteOffset: headerSize, as: UInt32.self)
    }
    let triangleRecordSize = 50
    let expectedSize = headerSize + countSize + Int(triCount) * triangleRecordSize
    guard data.count >= expectedSize, triCount > 0 else { return nil }

    var vertices: [SCNVector3] = []
    vertices.reserveCapacity(Int(triCount) * 3)
    var indices: [Int32] = []
    indices.reserveCapacity(Int(triCount) * 3)

    data.withUnsafeBytes { raw in
        var offset = headerSize + countSize
        for t in 0..<Int(triCount) {
            // Skip the 12-byte normal; read the 3 vertices (9 floats).
            let vertexOffset = offset + 12
            for v in 0..<3 {
                let base = vertexOffset + v * 12
                let x = raw.loadUnaligned(fromByteOffset: base, as: Float32.self)
                let y = raw.loadUnaligned(fromByteOffset: base + 4, as: Float32.self)
                let z = raw.loadUnaligned(fromByteOffset: base + 8, as: Float32.self)
                // STL vertices are in this app's established VTK (x,y,z)
                // surface-export order, the reverse of its own [z,y,x]
                // world-mm convention — same swap mmToScene applies
                // elsewhere (see su2_export.py's _xyz() for why).
                vertices.append(SCNVector3(x, z, y))
                indices.append(Int32(t * 3 + v))
            }
            offset += triangleRecordSize
        }
    }

    let source = SCNGeometrySource(vertices: vertices)
    let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
    let element = SCNGeometryElement(
        data: indexData, primitiveType: .triangles,
        primitiveCount: indices.count / 3, bytesPerIndex: MemoryLayout<Int32>.size
    )
    let geometry = SCNGeometry(sources: [source], elements: [element])
    let material = SCNMaterial()
    material.diffuse.contents = NSColor(red: 0.85, green: 0.25, blue: 0.25, alpha: 1.0)
    material.lightingModel = .physicallyBased
    material.isDoubleSided = true
    geometry.materials = [material]
    return geometry
}

/// world (x,y,z) mm, in this app's established *surface* convention (see
/// su2_export.py's own _xyz() docstring — the reverse of the app's other
/// [z,y,x] world-mm convention) -> SceneKit (x,z,y), matching exactly how
/// parseBinarySTL's own inline swap places every other surface this app
/// renders into the same scene space. The mesh-geometry endpoint's own
/// centroid_mm/at_mm values come straight off the same .vtp this STL was
/// converted from, so they're in that same surface convention too.
private func surfaceXYZToScene(_ p: [Double]) -> SCNVector3 {
    SCNVector3(Float(p[0]), Float(p[2]), Float(p[1]))
}

/// One diameter/angle callout anchored to a real point on the mesh (a
/// cap's centroid, or an angle's own "at_mm" midpoint) plus the offset
/// point its label text should float at.
struct MeshAnnotationAnchor {
    let id: String
    let anchorMM: SCNVector3
    let labelMM: SCNVector3
    let text: String
}

/// Published every frame by SurfacePreviewCoordinator's renderer-delegate
/// callback — each annotation's *current* screen-space anchor/label
/// position, re-projected through whatever the camera is doing right
/// now. SCNText was tried first and dropped: at this mesh's real-world
/// (mm) scale its glyphs either disappear into sub-pixel geometry or need
/// per-view scale-fudging to stay legible, and still comes out jagged —
/// re-projecting a crisp SwiftUI Text's position every frame instead
/// keeps the label glued to the right spot through orbit/zoom/pan without
/// any of that.
@MainActor
final class MeshAnnotationOverlayModel: ObservableObject {
    struct Projected {
        let anchor: CGPoint
        let label: CGPoint
        let text: String
        let visible: Bool
    }
    @Published var projected: [String: Projected] = [:]
}

@MainActor
final class SurfacePreviewCoordinator: NSObject, SCNSceneRendererDelegate {
    var cameraNode: SCNNode?
    var surfaceNode: SCNNode?
    let dragHandler = OrbitCameraDragHandler()

    // Annotation live-projection (mesh preview only, nil elsewhere) —
    // see MeshAnnotationOverlayModel's own doc comment for why this
    // re-projects every frame instead of placing text once.
    weak var scnView: SCNView?
    var annotationAnchors: [MeshAnnotationAnchor] = []
    weak var overlayModel: MeshAnnotationOverlayModel?

    nonisolated func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        DispatchQueue.main.async { [weak self] in
            self?.updateProjections()
        }
    }

    @MainActor
    private func updateProjections() {
        guard let scnView, !annotationAnchors.isEmpty else { return }
        var next: [String: MeshAnnotationOverlayModel.Projected] = [:]
        for a in annotationAnchors {
            let anchorP = scnView.projectPoint(a.anchorMM)
            let labelP = scnView.projectPoint(a.labelMM)
            // projectPoint's z is normalized device depth — outside
            // [0, 1] means behind the camera or past the far plane, not
            // actually on screen, so hide rather than draw a label at
            // some meaningless projected position.
            let visible = anchorP.z > 0 && anchorP.z < 1 && labelP.z > 0 && labelP.z < 1
            next[a.id] = .init(
                anchor: CGPoint(x: CGFloat(anchorP.x), y: CGFloat(anchorP.y)),
                label: CGPoint(x: CGFloat(labelP.x), y: CGFloat(labelP.y)),
                text: a.text, visible: visible
            )
        }
        overlayModel?.projected = next
    }
}

/// Renders a surface while preserving its camera between updates. STL
/// previews stay fixed; CFD results can replace the color source
/// while retaining the camera and the mesh's vertex/index sources.
struct SurfaceGeometryView: NSViewRepresentable {
    let geometry: SCNGeometry
    var controller: Scene3DController?
    // Mesh preview only (nil for the plain Build Surface STL preview) —
    // drives the diameter/angle callouts built in makeNSView below.
    var meshGeometry: APIClient.SU2MeshGeometry?
    var annotationOverlay: MeshAnnotationOverlayModel?

    func makeCoordinator() -> SurfacePreviewCoordinator { SurfacePreviewCoordinator() }

    func makeNSView(context: Context) -> Interactive3DSceneView {
        let view = Interactive3DSceneView()
        view.axisTag = "surface-preview"
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

        guard let scene = view.scene else { return view }
        let node = SCNNode(geometry: geometry)
        context.coordinator.surfaceNode = node
        scene.rootNode.addChildNode(node)

        let (center, radiusFloat) = node.boundingSphere
        let radius = CGFloat(max(radiusFloat, 1))
        controller?.attach(view: view, center: center, radius: radius)

        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.zFar = Double(radius) * 10 + 100
        cameraNode.camera = camera
        let distance = radius * 2.5 + 10
        cameraNode.position = SCNVector3(center.x, center.y, center.z + distance)
        cameraNode.look(at: center)
        context.coordinator.cameraNode = cameraNode
        scene.rootNode.addChildNode(cameraNode)
        view.pointOfView = cameraNode
        view.defaultCameraController.target = center

        if let meshGeometry {
            let anchors = Self.buildAnnotations(meshGeometry: meshGeometry, meshCenter: center, meshRadius: radius, into: scene)
            context.coordinator.annotationAnchors = anchors
            context.coordinator.scnView = view
            context.coordinator.overlayModel = annotationOverlay
            view.delegate = context.coordinator
        }

        return view
    }

    func updateNSView(_ nsView: Interactive3DSceneView, context: Context) {
        if let node = context.coordinator.surfaceNode, node.geometry !== geometry {
            node.geometry = geometry
            nsView.needsDisplay = true
        }
    }

    /// Builds the real 3D leader-line + anchor-dot geometry for every
    /// cap diameter and inter-cap angle the mesh-geometry endpoint
    /// reported, adding it straight into `scene`, and returns the
    /// matching (anchor, label) 3D positions for the coordinator to
    /// re-project into screen space every frame (see
    /// MeshAnnotationOverlayModel).
    private static func buildAnnotations(
        meshGeometry: APIClient.SU2MeshGeometry, meshCenter: SCNVector3, meshRadius: CGFloat, into scene: SCNScene
    ) -> [MeshAnnotationAnchor] {
        var anchors: [MeshAnnotationAnchor] = []
        // Kept well under the mesh's own radius — this app's meshes are
        // long thin tubes, not round blobs, so a cap sitting near one
        // end of the bounding *sphere* has much less spare margin to its
        // own camera-frustum edge than a sphere's radius alone suggests
        // (CONFIRMED: an 0.55-radius push sent the CCA inlet label
        // straight off the top of the preview). A smaller, mostly
        // direction-driven offset reads just as clearly without that risk.
        let labelOffsetMM = Float(max(meshRadius * 0.3, 6))
        let annotationsNode = SCNNode()

        /// Offsets `anchor` outward along `direction` (falling back to
        /// "away from the mesh's own center" when no better direction is
        /// known, e.g. for an angle's shared midpoint) by the shared
        /// label offset.
        func labelPosition(for anchor: SCNVector3, direction: SCNVector3?) -> SCNVector3 {
            let d = direction ?? SCNVector3(anchor.x - meshCenter.x, anchor.y - meshCenter.y, anchor.z - meshCenter.z)
            let len = sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
            guard len > 0.001 else { return SCNVector3(anchor.x, anchor.y + CGFloat(labelOffsetMM), anchor.z) }
            let scale = CGFloat(labelOffsetMM) / len
            return SCNVector3(anchor.x + d.x * scale, anchor.y + d.y * scale, anchor.z + d.z * scale)
        }

        func addLeader(anchor: SCNVector3, label: SCNVector3) {
            annotationsNode.addChildNode(lineNode(points: [anchor, label], color: .white))
            let dot = SCNSphere(radius: 0.5)
            dot.firstMaterial?.diffuse.contents = NSColor.white
            dot.firstMaterial?.lightingModel = .constant
            let dotNode = SCNNode(geometry: dot)
            dotNode.position = anchor
            annotationsNode.addChildNode(dotNode)
        }

        let capTitles: [(tag: String, title: String)] = [
            ("cca_inlet", "CCA inlet"), ("ica_outlet", "ICA outlet"), ("eca_outlet", "ECA outlet"),
        ]
        for (tag, title) in capTitles {
            guard let cap = meshGeometry.caps[tag] else { continue }
            let anchor = surfaceXYZToScene(cap.centroidMm)
            // The cap's own outward normal is a far more reliable "point
            // the label away from the vessel" direction than "away from
            // the whole mesh's center" — it's always locally correct
            // regardless of how long/bent the rest of the vessel is.
            let direction = cap.normalMm.map(surfaceXYZToScene)
            let label = labelPosition(for: anchor, direction: direction)
            addLeader(anchor: anchor, label: label)
            anchors.append(MeshAnnotationAnchor(
                id: "cap:\(tag)", anchorMM: anchor, labelMM: label,
                text: String(format: "%@\nØ %.2f mm", title, cap.equivalentDiameterMm)
            ))
        }
        for angle in meshGeometry.angles {
            let anchor = surfaceXYZToScene(angle.atMm)
            let label = labelPosition(for: anchor, direction: nil)
            addLeader(anchor: anchor, label: label)
            let niceLabel = angle.label
                .replacingOccurrences(of: "_outlet", with: "")
                .replacingOccurrences(of: "_inlet", with: "")
                .replacingOccurrences(of: "_", with: " ")
                .uppercased()
            anchors.append(MeshAnnotationAnchor(
                id: "angle:\(angle.label)", anchorMM: anchor, labelMM: label,
                text: String(format: "%@\n%.1f°", niceLabel, angle.degrees)
            ))
        }

        scene.rootNode.addChildNode(annotationsNode)
        return anchors
    }
}

/// SwiftUI half of the mesh-preview callouts — one white pill-background
/// label per visible annotation, floated at its live-projected screen
/// position (the real leader line + anchor dot are 3D SceneKit geometry
/// built alongside the mesh itself, see SurfaceGeometryView.buildAnnotations).
private struct MeshAnnotationOverlayView: View {
    @ObservedObject var model: MeshAnnotationOverlayModel

    var body: some View {
        ForEach(model.projected.keys.sorted(), id: \.self) { key in
            if let p = model.projected[key], p.visible {
                Text(p.text)
                    .font(Theme.labelFont(11, weight: .semibold))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .fixedSize()
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .background(RoundedRectangle(cornerRadius: 5).fill(Color.black.opacity(0.6)))
                    .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Color.white.opacity(0.35), lineWidth: 1))
                    .position(x: p.label.x, y: p.label.y)
            }
        }
    }
}

/// What SurfacePreviewView is previewing — the fast Build Surface STL
/// (see server/build_surface.py), or a just-built SU2 mesh's own
/// boundary surface (fetched as STL via a dedicated backend conversion
/// endpoint purely for this preview, since the mesh's real saved format
/// is .vtu/.vtp, not .stl — see server/app.py's su2_mesh/preview_stl).
/// Both are triangulated surfaces in the same binary-STL shape once
/// fetched, so one view/parser serves both; only where the bytes come
/// from and what "Save" does differs.
enum SurfacePreviewSource {
    case buildSurface
    case su2Mesh
}

/// Shown before committing to "Save STL…"/"Save Mesh…" — lets the just-
/// built geometry be looked at and orbited in 3D first, catching an
/// obviously bad build (the kind of artifact this session spent a while
/// chasing: corrugation, spikes, a surface cut shorter than the
/// segmented path — or, for a mesh, a collapsed/degenerate region) before
/// writing it to disk, rather than only finding out after opening the
/// file in a separate viewer.
struct SurfacePreviewView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    let side: String
    var source: SurfacePreviewSource = .buildSurface

    @State private var geometry: SCNGeometry?
    @State private var isLoading = true
    @State private var loadError: String?
    @State private var meshGeometry: APIClient.SU2MeshGeometry?
    @StateObject private var controller = Scene3DController()
    @StateObject private var annotationOverlay = MeshAnnotationOverlayModel()

    private var title: String {
        switch source {
        case .buildSurface: return "Surface Preview — \(side.capitalized)"
        case .su2Mesh: return "Mesh Preview — \(side.capitalized)"
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }

            ZStack {
                Color.black
                if let geometry {
                    SurfaceGeometryView(
                        geometry: geometry, controller: controller,
                        meshGeometry: meshGeometry, annotationOverlay: annotationOverlay
                    )
                    .overlay(alignment: .topTrailing) {
                        Scene3DControlsOverlay(controller: controller)
                    }
                    .overlay {
                        MeshAnnotationOverlayView(model: annotationOverlay)
                            .allowsHitTesting(false)
                    }
                } else if isLoading {
                    ProgressView("Loading…")
                        .foregroundStyle(.white)
                } else if let loadError {
                    Text(loadError)
                        .foregroundStyle(.red)
                        .padding()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                Spacer()
                switch source {
                case .buildSurface:
                    Button("Save STL…") { saveSurface() }
                        .buttonStyle(.borderedProminent)
                        .disabled(geometry == nil)
                case .su2Mesh:
                    Button("Save Mesh…") { saveMesh() }
                        .buttonStyle(.borderedProminent)
                        .disabled(geometry == nil)
                }
            }
        }
        .padding()
        .frame(minWidth: 700, minHeight: 600)
        .task {
            await loadSurface()
        }
    }

    private func loadSurface() async {
        isLoading = true
        defer { isLoading = false }
        guard let caseId = state.caseInfo?.caseId else {
            loadError = "No case loaded."
            return
        }
        do {
            let data: Data
            switch source {
            case .buildSurface:
                data = try await state.api.fetchMeshSTL(caseId: caseId, side: side)
            case .su2Mesh:
                data = try await state.api.fetchSU2MeshPreviewSTL(caseId: caseId, side: side)
                // Best-effort — a missing/failed geometry fetch just
                // means no diameter/angle callouts, not a reason to fail
                // the whole preview (the mesh itself is still useful to
                // look at on its own).
                meshGeometry = try? await state.api.fetchSU2MeshGeometry(caseId: caseId, side: side)
            }
            guard let parsed = parseBinarySTL(data) else {
                loadError = "Couldn't parse the STL — it may be empty or malformed."
                return
            }
            geometry = parsed
        } catch {
            loadError = "Failed to load: \(error.localizedDescription)"
        }
    }

    private func saveSurface() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "stl")!]
        panel.nameFieldStringValue = "\(side)_carotid.stl"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await state.saveSurfaceSTL(side: side, to: url) }
    }

    private func saveMesh() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Save Mesh Here"
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        let vtuURL = folder.appendingPathComponent("\(side)_carotid_mesh.vtu")
        let vtpURL = folder.appendingPathComponent("\(side)_carotid_mesh_surface.vtp")
        Task { await state.saveSU2Mesh(side: side, vtuURL: vtuURL, vtpURL: vtpURL) }
    }
}
