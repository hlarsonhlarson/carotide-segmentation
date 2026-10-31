import SceneKit
import SwiftUI

/// Lets a plain SwiftUI button (outside the NSViewRepresentable) drive a
/// 3D preview's camera — "recenter on content" — without having to
/// round-trip through SwiftUI's own Coordinator mechanism. The owning
/// view's rebuildScene re-attaches the current SCNView + content bounds
/// every time it rebuilds, so recenter always acts on up-to-date bounds
/// even as the wall preview grows.
@MainActor
final class Scene3DController: ObservableObject {
    private weak var view: SCNView?
    private var boundingCenter: SCNVector3 = SCNVector3Zero
    private var boundingRadius: CGFloat = 50

    func attach(view: SCNView, center: SCNVector3, radius: CGFloat) {
        self.view = view
        self.boundingCenter = center
        self.boundingRadius = max(radius, 1)
    }

    /// Resets to the same framing used when the view first appeared —
    /// doubles as a simple "reset view" when orbiting has wandered off
    /// and the content is hard to find again.
    func recenter() {
        guard let view, let pov = view.pointOfView else { return }
        let distance = boundingRadius * 2.5 + 10
        pov.position = SCNVector3(boundingCenter.x, boundingCenter.y, boundingCenter.z + distance)
        pov.look(at: boundingCenter)
        view.defaultCameraController.target = boundingCenter
    }
}

/// Small corner overlay shared by every 3D preview pane in the app.
struct Scene3DControlsOverlay: View {
    @ObservedObject var controller: Scene3DController

    var body: some View {
        Button { controller.recenter() } label: {
            Image(systemName: "scope")
        }
        .help("Recenter")
        .buttonStyle(.bordered)
        .controlSize(.small)
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.35)))
        .padding(8)
    }
}
