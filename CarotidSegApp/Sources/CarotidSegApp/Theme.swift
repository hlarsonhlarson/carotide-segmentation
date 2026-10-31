import SwiftUI

/// This app's own flat visual language — sampled directly (pixel colors,
/// not eyeballed) from a recording of the actual original Colin McRae
/// Rally 2.0 UI (its main menu, tag entry, and country-select screens),
/// which the user picked as the reference after an earlier attempt based
/// on a fan-made redesign missed the mark (too light, a saturated yellow
/// accent that original never has). The real thing is much starker: one
/// dark desaturated blue-grey background, no accent color anywhere, and
/// emphasis (what's currently selected vs. everything else) carried
/// entirely by brightness and weight — a dim grey-blue label versus a
/// near-white bold one, nothing hue-based.
enum Theme {
    /// Primary window/pane background — the reference's own menu
    /// background, sampled directly from the recording.
    static let background = Color(red: 68/255, green: 87/255, blue: 111/255)
    /// A darker recessed panel — the reference's own bottom nav strip is
    /// a few shades darker than the main background; reused here for
    /// nested panels (debug readouts, list rows) that need to read as a
    /// distinct surface without any drop shadow (the reference has none).
    static let panel = Color(red: 52/255, green: 65/255, blue: 82/255)
    /// A lighter raised surface — the inverse move, for a row or button
    /// that should read as sitting *above* the background (a hovered /
    /// selected list row, a button fill).
    static let panelRaised = Color(red: 88/255, green: 107/255, blue: 131/255)
    /// Primary text and "this is the currently-relevant one" emphasis —
    /// the reference's own near-white, used for the active breadcrumb
    /// item, selected menu entry, and every numeric readout.
    static let textBright = Color(red: 239/255, green: 242/255, blue: 247/255)
    /// Secondary/inactive text — the reference's own dimmer blue-grey,
    /// used for everything not currently selected or relevant.
    static let textDim = Color(red: 153/255, green: 172/255, blue: 196/255)

    // Old names kept as aliases during the transition away from the
    // light sage palette, so call sites didn't all need renaming in the
    // same pass as the recolor — `ink`/`inkMuted` now point at the
    // bright/dim text colors a dark background actually needs.
    static let ink = textBright
    static let inkMuted = textDim

    /// The reference's own numeric displays (lap time, countdown) are
    /// plain bold text, not a separate digital/7-segment face — reused
    /// here for this app's own numeric readouts.
    static func readoutFont(_ size: CGFloat = 13, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    /// Heading/label face — the reference uses a clean, fairly heavy
    /// grotesque sans throughout, not a rounded/friendly one; the plain
    /// system (SF Pro) design reads much closer to that than .rounded
    /// did.
    static func labelFont(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default)
    }
}

/// A panel surface in this theme: `Theme.panel` fill, no border/shadow —
/// the reference has no elevation system, every surface is flat.
struct ThemedPanel: ViewModifier {
    var padding: CGFloat = 10
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(RoundedRectangle(cornerRadius: 6).fill(Theme.panel))
    }
}

extension View {
    func themedPanel(padding: CGFloat = 10) -> some View {
        modifier(ThemedPanel(padding: padding))
    }
}

/// The reference's own numeric-readout block: a darker recessed
/// background, bright bold digits — used for this app's own numeric
/// displays (slice index, point counts) in place of a plain Text.
struct ReadoutLabel: View {
    let value: String
    var caption: String?
    var size: CGFloat = 15

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(Theme.readoutFont(size))
                .foregroundStyle(Theme.textBright)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(RoundedRectangle(cornerRadius: 4).fill(Theme.panel))
            if let caption {
                Text(caption)
                    .font(Theme.labelFont(10))
                    .foregroundStyle(Theme.textDim)
            }
        }
    }
}

/// A button in this theme. The reference carries emphasis with
/// brightness, not hue — `isPrimary` fills with the bright near-white
/// (dark text on top) the same way the reference's own selected menu
/// entry turns bright white while everything else around it stays a
/// dim blue-grey; the default state is the raised-panel fill instead.
struct ThemedButtonStyle: ButtonStyle {
    var isPrimary: Bool = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Theme.labelFont(12, weight: .semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill(isPrimary ? Theme.textBright : Theme.panelRaised)
                    .opacity(configuration.isPressed ? 0.7 : 1)
            )
            .foregroundStyle(isPrimary ? Theme.background : Theme.textBright)
    }
}

extension ButtonStyle where Self == ThemedButtonStyle {
    static var themed: ThemedButtonStyle { ThemedButtonStyle() }
    static var themedPrimary: ThemedButtonStyle { ThemedButtonStyle(isPrimary: true) }
}

/// A high-contrast progress bar for this theme — the plain system
/// `ProgressView(value:)` only tints its *fill*, leaving its own
/// unfilled track a faint translucent grey that all but disappears
/// against this app's own mid-tone blue-grey background (CONFIRMED: a
/// long CFD run's own progress bar was reported as "not visible" for
/// exactly this reason). A dark recessed track (the same fill
/// `ThemedPanel` uses for every other recessed surface) under a solid
/// near-white fill keeps both the done and remaining portions legible at
/// a glance, matching the reference's own brightness-only emphasis.
struct ThemedProgressBar: View {
    /// 0...1 — callers already clamp their own fractionComplete, but a
    /// stray out-of-range value shouldn't draw a fill past the track's
    /// own bounds or into negative width.
    var fraction: Double
    var height: CGFloat = 10

    var body: some View {
        GeometryReader { geo in
            let clamped = min(max(fraction, 0), 1)
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: height / 2)
                    .fill(Theme.panel)
                RoundedRectangle(cornerRadius: height / 2)
                    .fill(Theme.textBright)
                    .frame(width: geo.size.width * clamped)
            }
        }
        .frame(height: height)
        .overlay(
            RoundedRectangle(cornerRadius: height / 2)
                .strokeBorder(Theme.textDim.opacity(0.4), lineWidth: 1)
        )
    }
}
