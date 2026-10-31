import SwiftUI
import UniformTypeIdentifiers

/// Side panel listing every placed control point for the current side,
/// one column per path (CCA→ICA, CCA→ECA), each colored to match that
/// path's color everywhere else in the app (branchColor) — with a
/// delete button per point, since the 2D-pane click-to-delete (double-
/// click a marker) isn't always precise enough to pick one of several
/// close-together points, and there's no way at all to see the raw
/// coordinates without this.
struct CenterlinePointsPanel: View {
    @EnvironmentObject var state: AppState
    private let branches = ["cca_ica", "cca_eca"]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("PATH POINTS — \(state.selectedSide.uppercased())")
                .font(Theme.labelFont(11, weight: .bold))
                .foregroundStyle(Theme.textDim)

            ScrollView {
                HStack(alignment: .top, spacing: 10) {
                    ForEach(branches, id: \.self) { branch in
                        column(for: branch)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(10)
        .frame(width: 320)
        .frame(maxHeight: .infinity)
        .background(Theme.background)
    }

    private func column(for branch: String) -> some View {
        let key = state.centerlineKey(side: state.selectedSide, branch: branch)
        let points = state.controlPoints[key] ?? []
        let color = branchColor(branch)

        return VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 5) {
                Circle().fill(color).frame(width: 7, height: 7)
                Text(branchDisplayName(branch).uppercased())
                    .font(Theme.labelFont(11, weight: .bold))
                    .foregroundStyle(Theme.textBright)
            }
            .padding(.bottom, 2)

            if points.isEmpty {
                Text("—")
                    .font(Theme.labelFont(12))
                    .foregroundStyle(Theme.textDim)
            } else {
                ForEach(Array(points.enumerated()), id: \.offset) { idx, voxel in
                    let isBeingDragged = state.reorderingPointKey == key && state.reorderingPointIndex == idx
                    let isDropTarget = state.reorderDropTargetKey == key && state.reorderDropTargetIndex == idx

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 5) {
                            ZStack {
                                Circle().fill(color)
                                Text("\(idx)")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(Theme.background)
                            }
                            .frame(width: 18, height: 18)

                            Spacer(minLength: 2)

                            Button {
                                let side = state.selectedSide
                                Task { await state.deleteCenterlinePoint(side: side, branch: branch, index: idx) }
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 12))
                                    .foregroundStyle(Theme.textDim)
                            }
                            .buttonStyle(.plain)
                        }

                        // On its own full-width line (not sharing the row
                        // with the badge/delete icons) so three-digit voxel
                        // coordinates never get squeezed or truncated in a
                        // ~90pt-wide column. Always the theme's own bright
                        // text, never the raw branch color — CONFIRMED
                        // (against the previous light background) that a
                        // branch color picked for contrast against the 2D
                        // slice panes' own black imagery can still turn
                        // unreadable against a differently-colored panel
                        // background elsewhere; this stays legible
                        // regardless of what branchColor returns.
                        Text("\(voxel[0]), \(voxel[1]), \(voxel[2])")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(Theme.textBright)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 7)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(isDropTarget ? Theme.panelRaised : Theme.panel)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(isDropTarget ? Theme.textBright : color.opacity(0.5), lineWidth: isDropTarget ? 2 : 1)
                    )
                    .opacity(isBeingDragged ? 0.45 : 1.0)
                    .contentShape(Rectangle())
                    .onDrag {
                        state.reorderingPointKey = key
                        state.reorderingPointIndex = idx
                        return NSItemProvider(object: "\(branch)|\(idx)" as NSString)
                    }
                    .onDrop(of: [.text], delegate: PointReorderDropDelegate(
                        branch: branch, targetIndex: idx, side: state.selectedSide, key: key, state: state
                    ))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Backs the drag-to-reorder rows: drop payload is "branch|fromIndex" (set
/// by the dragged row's .onDrag), checked against this target row's own
/// branch/index so a drop only reorders within the same branch's column —
/// dragging a point onto a different branch's list wouldn't have a
/// sensible meaning here (each branch is its own independent path).
private struct PointReorderDropDelegate: DropDelegate {
    let branch: String
    let targetIndex: Int
    let side: String
    let key: String
    let state: AppState

    func performDrop(info: DropInfo) -> Bool {
        state.reorderingPointKey = nil
        state.reorderingPointIndex = nil
        state.reorderDropTargetKey = nil
        state.reorderDropTargetIndex = nil
        guard let provider = info.itemProviders(for: [.text]).first else { return false }
        provider.loadObject(ofClass: NSString.self) { reading, _ in
            guard let str = reading as? String else { return }
            let parts = str.split(separator: "|")
            guard parts.count == 2, parts[0] == branch, let fromIndex = Int(parts[1]), fromIndex != targetIndex else { return }
            Task { @MainActor in
                await state.reorderCenterlinePoint(side: side, branch: branch, fromIndex: fromIndex, toIndex: targetIndex)
            }
        }
        return true
    }

    func dropEntered(info: DropInfo) {
        state.reorderDropTargetKey = key
        state.reorderDropTargetIndex = targetIndex
    }

    func dropExited(info: DropInfo) {
        if state.reorderDropTargetKey == key && state.reorderDropTargetIndex == targetIndex {
            state.reorderDropTargetKey = nil
            state.reorderDropTargetIndex = nil
        }
    }

    func validateDrop(info: DropInfo) -> Bool {
        info.hasItemsConforming(to: [.text])
    }
}
