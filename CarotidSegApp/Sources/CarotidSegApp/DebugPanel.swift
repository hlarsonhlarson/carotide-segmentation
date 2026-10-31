import SwiftUI

/// Raw diagnostics from the last "Build Surface" call and the current/last
/// "Build SU2 Mesh" job — every field each backend diag dict returned
/// (branches_used, bbox_margin_mm, per-branch points_stamped/
/// substituted_outlier, n_boundary_loops, n_nonmanifold_edges,
/// n_surface_points/cells, n_volume_nodes/elements, face_id_counts, ...),
/// not just the handful AppState's own typed results name — so a
/// surface- or mesh-quality issue (an unusual number of substituted/
/// outlier points, an unexpected boundary-loop count, a suspiciously low
/// triangle/tet count) is visible right in the app instead of needing a
/// separate inspection pass each time. The SU2 Mesh section updates live
/// while a mesh job is running (see AppState.startSU2Mesh), not just once
/// it finishes.
struct DebugPanel: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    section(title: "BUILD SURFACE") {
                        if let result = state.lastBuildResult {
                            Text(result.rawJSON)
                                .foregroundStyle(result.ok ? Theme.ink : Color.red)
                        } else {
                            Text("Build a surface to see its diagnostics here.")
                                .foregroundStyle(Theme.inkMuted)
                        }
                    }

                    Divider().background(Theme.ink.opacity(0.15))

                    section(title: "SU2 MESH (\(state.su2MeshState.uppercased()))") {
                        if let json = state.su2MeshStatusJSON {
                            Text(json)
                                .foregroundStyle(state.su2MeshState == "error" ? Color.red : Theme.ink)
                        } else {
                            Text("Build a mesh to see live progress and diagnostics here.")
                                .foregroundStyle(Theme.inkMuted)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .themedPanel()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    @ViewBuilder
    private func section(title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(Theme.labelFont(10, weight: .bold))
                .foregroundStyle(Theme.inkMuted)
            content()
                .font(Theme.readoutFont(10, weight: .regular))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
