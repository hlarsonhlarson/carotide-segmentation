import SwiftUI

/// The six stages this app's workflow breaks into — the first four are
/// SimVascular's own vascular-modeling pipeline shape (Path Planning ->
/// Segmentation -> Model/Lofting -> Meshing); Boundary Conditions and CFD
/// Computation are this app's own addition on top: the per-patient
/// physiological inputs (CCA inlet waveform, ICA/ECA flow split, outlet
/// pressure) an SU2 run needs once the mesh itself is ready, then the
/// actual pulsatile solve those inputs drive.
enum WorkflowStep: Int, CaseIterable {
    case pathway = 1, segmentation, surface, meshing, boundaryConditions, cfdComputation

    var title: String {
        switch self {
        case .pathway: return "Pathway"
        case .segmentation: return "Segmentation"
        case .surface: return "Surface"
        case .meshing: return "Meshing"
        case .boundaryConditions: return "Boundary Conditions"
        case .cfdComputation: return "CFD Computation"
        }
    }

    var subtitle: String {
        switch self {
        case .pathway: return "Mark CCA→ICA and CCA→ECA paths"
        case .segmentation: return "Contour the vessel wall"
        case .surface: return "Loft into a 3D surface"
        case .meshing: return "Build the CFD-ready mesh"
        case .boundaryConditions: return "Set inlet/outlet flow parameters"
        case .cfdComputation: return "Run the pulsatile CFD solve"
        }
    }
}

/// A small vertical checklist showing which of the four stages the
/// currently-selected side is at — not a gate on navigation (every step
/// stays reachable regardless), just orientation. Each row also has its
/// own Save button, persisting that step's own data (for the currently
/// selected side) into the case's project folder — see
/// AppState.save*ToProject / chooseProjectFolder.
struct WorkflowStepsPanel: View {
    @EnvironmentObject var state: AppState
    let currentStep: WorkflowStep
    @State private var savingStep: WorkflowStep?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("WORKFLOW")
                    .font(Theme.labelFont(11, weight: .bold))
                    .foregroundStyle(Theme.inkMuted)
                Spacer()
                if let projectDir = state.projectDir {
                    Text(URL(fileURLWithPath: projectDir).lastPathComponent)
                        .font(Theme.readoutFont(10, weight: .medium))
                        .foregroundStyle(Theme.inkMuted)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(projectDir)
                } else {
                    Button("Choose Project Folder…") {
                        Task { await state.chooseProjectFolder() }
                    }
                    .buttonStyle(.themed)
                }
            }
            Divider().background(Theme.ink.opacity(0.2))

            ForEach(WorkflowStep.allCases, id: \.rawValue) { step in
                VStack(spacing: 0) {
                    HStack(alignment: .top, spacing: 8) {
                        ZStack {
                            Circle()
                                .fill(badgeColor(for: step))
                                .frame(width: 20, height: 20)
                            if step.rawValue < currentStep.rawValue {
                                Image(systemName: "checkmark")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundStyle(Theme.background)
                            } else {
                                Text("\(step.rawValue)")
                                    .font(Theme.readoutFont(10, weight: .bold))
                                    .foregroundStyle(step == currentStep ? Theme.background : Theme.textDim)
                            }
                        }
                        VStack(alignment: .leading, spacing: 1) {
                            Text(step.title)
                                .font(Theme.labelFont(12, weight: step == currentStep ? .bold : .medium))
                                .foregroundStyle(Theme.ink)
                            Text(step.subtitle)
                                .font(Theme.labelFont(10))
                                .foregroundStyle(Theme.inkMuted)
                        }
                        Spacer()
                        Button {
                            save(step)
                        } label: {
                            if savingStep == step {
                                ProgressView().controlSize(.small)
                            } else {
                                Text("Save")
                            }
                        }
                        .buttonStyle(.themed)
                        .disabled(savingStep != nil)
                    }
                    .padding(.vertical, 8)
                    if step != WorkflowStep.allCases.last {
                        Divider().background(Theme.ink.opacity(0.15))
                    }
                }
            }
        }
        .themedPanel()
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// No accent color, by design — matching the reference, which marks
    /// "current" purely by brightness (same bright fill as a completed
    /// step) rather than a third hue; the checkmark vs. digit inside is
    /// what actually tells completed and current apart.
    private func badgeColor(for step: WorkflowStep) -> Color {
        if step.rawValue <= currentStep.rawValue { return Theme.textBright }
        return Theme.panelRaised
    }

    private func save(_ step: WorkflowStep) {
        let side = state.selectedSide
        savingStep = step
        Task {
            switch step {
            case .pathway: await state.saveCenterlineToProject(side: side)
            case .segmentation: await state.saveSegmentationToProject(side: side)
            case .surface: await state.saveSurfaceToProject(side: side)
            case .meshing: await state.saveMeshToProject(side: side)
            case .boundaryConditions: await state.saveSU2ParamsToProject(side: side)
            case .cfdComputation: await state.saveCFDResultsToProject(side: side)
            }
            savingStep = nil
        }
    }
}
