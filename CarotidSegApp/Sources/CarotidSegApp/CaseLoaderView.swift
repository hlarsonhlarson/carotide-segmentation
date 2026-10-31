import SwiftUI
import UniformTypeIdentifiers

/// Which folder-picker action is currently open — CONFIRMED (two
/// separate `.fileImporter` modifiers chained on the same view, one for
/// the DICOM folder and one for a saved project) that SwiftUI on macOS
/// only wires up one of them; the other's isPresented binding flips true
/// but no panel ever appears and no error fires either, which from the
/// user's side just looks like a dead button. One shared `.fileImporter`
/// keyed off which purpose is active avoids the conflict outright
/// instead of needing to figure out which of the two modifiers SwiftUI
/// would silently drop.
///
/// `pickerPurpose` is intentionally a separate piece of state from the
/// importer's `isPresented` bool (`showPicker`), not derived from it —
/// an earlier version tied both to one optional and cleared it from the
/// `isPresented` binding's setter, which races against the completion
/// closure (macOS flips `isPresented` to false around the same time it
/// invokes the result handler, so `pickerPurpose` could already be nil
/// by the time the handler read it — every selection silently hit
/// `case nil: break`, with no request to the backend and no error).
/// Keeping `pickerPurpose` untouched by dismissal avoids that race.
private enum FolderPickerPurpose {
    case dicom, project
}

struct CaseLoaderView: View {
    @EnvironmentObject var state: AppState
    @State private var pickedFolder: URL?
    @State private var pickerPurpose: FolderPickerPurpose = .dicom
    @State private var showPicker = false
    @State private var availableSeries: [SeriesInfo] = []
    @State private var isListingSeries = false
    @State private var listError: String?

    var body: some View {
        VStack(spacing: 16) {
            Text("Carotid Segmentation")
                .font(Theme.labelFont(28, weight: .bold))
                .foregroundStyle(Theme.ink)
            Text("Open a DICOM study folder to begin, or reopen a saved project.")
                .font(Theme.labelFont(13))
                .foregroundStyle(Theme.inkMuted)
            HStack(spacing: 10) {
                Button("Choose DICOM Folder…") {
                    pickerPurpose = .dicom
                    showPicker = true
                }
                .buttonStyle(.themedPrimary)

                Button("Open Saved Project…") {
                    pickerPurpose = .project
                    showPicker = true
                }
                .buttonStyle(.themed)
            }

            if let folder = pickedFolder {
                Text(folder.path)
                    .font(Theme.readoutFont(11, weight: .regular))
                    .foregroundStyle(Theme.inkMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            if isListingSeries {
                ProgressView("Reading series…")
            }
            if let listError {
                Text(listError).foregroundStyle(.red)
            }
            if !availableSeries.isEmpty {
                VStack(spacing: 0) {
                    ForEach(availableSeries) { series in
                        Button {
                            Task {
                                await state.loadCase(path: pickedFolder!.path, seriesUID: series.seriesUID)
                            }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(series.description.isEmpty ? "(no description)" : series.description)
                                        .font(Theme.labelFont(13, weight: .semibold))
                                        .foregroundStyle(Theme.ink)
                                    Text("\(series.modality) · \(series.nFiles) file(s)")
                                        .font(Theme.labelFont(11))
                                        .foregroundStyle(Theme.inkMuted)
                                }
                                Spacer()
                            }
                            .padding(.vertical, 8)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        Divider().background(Theme.ink.opacity(0.15))
                    }
                }
                .themedPanel()
                .frame(minWidth: 420, minHeight: 320)
            }
            if state.isBusy {
                ProgressView("Loading case…")
            }
            if let error = state.errorMessage {
                Text(error).foregroundStyle(.red)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background)
        .fileImporter(
            isPresented: $showPicker,
            allowedContentTypes: [.folder]
        ) { result in
            switch result {
            case .success(let url):
                switch pickerPurpose {
                case .dicom:
                    pickedFolder = url
                    Task { await listSeries(at: url) }
                case .project:
                    Task { await state.openProject(path: url.path) }
                }
            case .failure(let error):
                listError = error.localizedDescription
            }
        }
    }

    private func listSeries(at url: URL) async {
        isListingSeries = true
        listError = nil
        availableSeries = []
        defer { isListingSeries = false }
        do {
            availableSeries = try await state.api.listSeries(path: url.path)
            if availableSeries.isEmpty {
                listError = "No DICOM series found in that folder."
            }
        } catch {
            listError = "Failed to list series: \(error.localizedDescription)"
        }
    }
}
