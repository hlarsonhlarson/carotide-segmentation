import Charts
import SwiftUI

/// Boundary Conditions step: the handful of physiological parameters that
/// actually vary patient to patient in the pysu2 pipeline this is modeled
/// on (~/Desktop/SU2 Pipeline Code/*/case_params.json) — CCA inlet
/// waveform (peak systolic / end-diastolic velocity, cardiac cycle
/// duration), the ICA/ECA outlet flow split, and the mean arterial
/// pressure level the Windkessel outlets are referenced to. Fluid
/// properties and solver numerics were constant across every real case
/// run there, so they aren't exposed here — this screen is deliberately
/// just the per-patient inputs, not a full SU2 config editor.
struct BoundaryConditionsView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    let side: String

    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Boundary Conditions — \(side.capitalized)")
                        .font(Theme.labelFont(18, weight: .bold))
                        .foregroundStyle(Theme.ink)
                    Text("CCA inlet waveform, ICA/ECA flow split, and outlet pressure level for the CFD run.")
                        .font(Theme.labelFont(11))
                        .foregroundStyle(Theme.inkMuted)
                }
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.themed)
            }

            if state.isLoadingSU2Params {
                ProgressView().controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 14) {
                waveformSection
                Divider().background(Theme.ink.opacity(0.15))
                flowSplitSection
                Divider().background(Theme.ink.opacity(0.15))
                pressureSection
            }
            .themedPanel(padding: 14)

            HStack {
                Spacer()
                Button {
                    save()
                } label: {
                    if isSaving {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Save")
                    }
                }
                .buttonStyle(.themedPrimary)
                .disabled(isSaving)
            }

            if let error = state.errorMessage {
                Text(error).foregroundStyle(.red)
                    .font(Theme.labelFont(11))
            }
        }
        .padding(20)
        .frame(minWidth: 460)
        .background(Theme.background)
        .task {
            await state.loadSU2Params(side: side)
        }
    }

    private var waveformSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("CCA INLET WAVEFORM")
                .font(Theme.labelFont(11, weight: .bold))
                .foregroundStyle(Theme.inkMuted)
            waveformChart
            paramRow(
                label: "Peak Systolic Velocity", unit: "cm/s",
                value: $state.su2Params.psvCms, range: 20...250
            )
            paramRow(
                label: "End-Diastolic Velocity", unit: "cm/s",
                value: $state.su2Params.edvCms, range: 0...150
            )
            paramRow(
                label: "Cardiac Cycle Duration", unit: "s",
                value: $state.su2Params.cardiacCycleS, range: 0.3...2.0
            )
        }
    }

    /// One cycle of the CCA inlet velocity curve this side's current
    /// PSV/EDV/cardiac-cycle settings would actually drive the SU2 solver
    /// with — the exact same shape as CCAWaveformShape mirrors from
    /// cca_waveform_psv_edv.py, just resampled live off the sliders'
    /// current (not-yet-saved) values rather than waiting for a Save or a
    /// backend round trip, since it's a pure local function of three
    /// numbers already held in state.
    private var waveformSamples: [(t: Double, v: Double)] {
        let cycle = max(state.su2Params.cardiacCycleS, 0.01)
        let psv = state.su2Params.psvCms
        let edv = state.su2Params.edvCms
        let n = 160
        return (0...n).map { i in
            let frac = Double(i) / Double(n)
            return (frac * cycle, CCAWaveformShape.velocityCms(atFraction: frac, psvCms: psv, edvCms: edv))
        }
    }

    private var waveformMeanCms: Double {
        let psv = state.su2Params.psvCms
        let edv = state.su2Params.edvCms
        let n = 400
        let total = (0..<n).reduce(0.0) { acc, i in
            acc + CCAWaveformShape.velocityCms(atFraction: Double(i) / Double(n), psvCms: psv, edvCms: edv)
        }
        return total / Double(n)
    }

    private var waveformChart: some View {
        VStack(alignment: .leading, spacing: 4) {
            Chart(waveformSamples, id: \.t) { sample in
                AreaMark(x: .value("Time", sample.t), y: .value("Velocity", sample.v))
                    .foregroundStyle(Theme.textBright.opacity(0.12))
                LineMark(x: .value("Time", sample.t), y: .value("Velocity", sample.v))
                    .foregroundStyle(Theme.textBright)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.catmullRom)
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                    AxisGridLine().foregroundStyle(Theme.ink.opacity(0.15))
                    AxisValueLabel().foregroundStyle(Theme.inkMuted)
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisGridLine().foregroundStyle(Theme.ink.opacity(0.15))
                    AxisValueLabel().foregroundStyle(Theme.inkMuted)
                }
            }
            .frame(height: 130)

            HStack {
                Text("time (s)")
                    .font(Theme.labelFont(9))
                    .foregroundStyle(Theme.inkMuted)
                Spacer()
                Text(String(format: "mean %.0f cm/s", waveformMeanCms))
                    .font(Theme.readoutFont(10))
                    .foregroundStyle(Theme.inkMuted)
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 5).fill(Theme.panel))
    }

    private var flowSplitSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("ICA / ECA FLOW SPLIT")
                .font(Theme.labelFont(11, weight: .bold))
                .foregroundStyle(Theme.inkMuted)
            paramRow(
                label: "ICA Fraction of Flow", unit: "",
                value: $state.su2Params.icaFraction, range: 0.05...0.95, step: 0.01
            )
            HStack {
                Text("ECA gets the remainder:")
                    .font(Theme.labelFont(11))
                    .foregroundStyle(Theme.inkMuted)
                ReadoutLabel(value: String(format: "%.0f%%", (1 - state.su2Params.icaFraction) * 100), size: 12)
            }
        }
    }

    private var pressureSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("OUTLET PRESSURE LEVEL")
                .font(Theme.labelFont(11, weight: .bold))
                .foregroundStyle(Theme.inkMuted)
            // Independently editable per outlet — the two outlets' own
            // downstream vascular beds don't necessarily sit at the same
            // operating pressure.
            paramRow(
                label: "ICA Outlet Pressure", unit: "mmHg",
                value: $state.su2Params.icaMapMmHg, range: 40...160
            )
            paramRow(
                label: "ECA Outlet Pressure", unit: "mmHg",
                value: $state.su2Params.ecaMapMmHg, range: 40...160
            )
        }
    }

    private func paramRow(label: String, unit: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double = 1) -> some View {
        HStack {
            Text(label)
                .font(Theme.labelFont(12))
                .foregroundStyle(Theme.ink)
                .frame(width: 190, alignment: .leading)
            Slider(value: value, in: range, step: step) { isEditing in
                // Pushed to the backend only once the drag ends, not on
                // every intermediate value while dragging — a Slider's
                // binding fires continuously mid-drag, and this is a
                // local HTTP round trip per update.
                if !isEditing {
                    Task { await state.updateSU2Params() }
                }
            }
            Text(unit.isEmpty ? String(format: "%.2f", value.wrappedValue) : String(format: "%.1f %@", value.wrappedValue, unit))
                .font(Theme.readoutFont(12))
                .foregroundStyle(Theme.textBright)
                .frame(width: 80, alignment: .trailing)
        }
    }

    private func save() {
        isSaving = true
        Task {
            await state.updateSU2Params()
            await state.saveSU2ParamsToProject(side: side)
            isSaving = false
        }
    }
}
