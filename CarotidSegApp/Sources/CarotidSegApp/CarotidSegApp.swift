import AppKit
import SwiftUI

final class AppActivator: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        bringMainWindowFront(attempt: 0)
    }

    // SwiftUI's WindowGroup window isn't guaranteed to exist yet when this
    // delegate method fires (observed on a real run: the app became the
    // active/frontmost app per the menu bar, but no window ever appeared —
    // NSApp.windows was empty at this point). Retry briefly on the next
    // run-loop turns instead of ordering front a window list that may still
    // be empty.
    private func bringMainWindowFront(attempt: Int) {
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.isVisible || $0.canBecomeKey }) {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
            return
        }
        guard attempt < 20 else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            self.bringMainWindowFront(attempt: attempt + 1)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        Group {
            if state.caseInfo != nil {
                SliceView()
            } else {
                CaseLoaderView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background)
        .toolbar {
            if state.caseInfo != nil {
                ToolbarItem {
                    Button("Open Different Case…") {
                        state.caseInfo = nil
                    }
                    .buttonStyle(.themed)
                }
            }
        }
        .onAppear {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }
}

@main
struct CarotidSegApp: App {
    @StateObject private var state = AppState()
    @NSApplicationDelegateAdaptor(AppActivator.self) var appDelegate

    init() {
        // stdout is fully buffered (not line-buffered) when it isn't a
        // tty — e.g. piped through `tee` by the Desktop launcher, which
        // is how debug prints get captured to a log file. Without this,
        // prints sit in the buffer and never reach the file until the
        // process exits, making live debugging impossible.
        setvbuf(stdout, nil, _IOLBF, 0)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(state)
                .frame(minWidth: 1020, minHeight: 760)
                // The app's own theme (see Theme.swift) is a dark
                // blue-grey palette sampled from the actual Colin McRae
                // Rally 2.0 UI — forcing .dark here makes every stock
                // control (sliders, pickers, toggles) render with
                // dark-mode chrome, matching it instead of clashing.
                .preferredColorScheme(.dark)
        }
    }
}
