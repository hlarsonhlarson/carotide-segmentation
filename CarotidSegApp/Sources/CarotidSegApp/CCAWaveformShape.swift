import Foundation

/// The same reconstructed CCA inlet waveform shape the real pysu2
/// pipeline drives its solver with (~/Desktop/SU2 Pipeline Code/shared/
/// cca_waveform_psv_edv.py) — a periodic cubic spline through literature-
/// calibrated landmark points (a reconstruction calibrated to Holdsworth
/// et al. 1999's published peak/notch timing and magnitude, not verbatim
/// digitized patient data — see that file's own docstring for the full
/// honesty note, which still applies here), normalized so its own cycle
/// mean is 1.0.
///
/// Precomputed here as a dense static lookup (201 points over one
/// normalized cycle, generated once from the original Python
/// scipy.interpolate.CubicSpline(..., bc_type="periodic")) rather than
/// re-deriving a periodic cubic spline in Swift — the shape itself is
/// fixed (only PSV/EDV/cycle-duration ever vary the rescaled curve, never
/// the shape), so a table this dense is visually indistinguishable from
/// the original spline and avoids needing a spline-fitting library here.
enum CCAWaveformShape {
    static let qMin = 0.4904478281171064
    static let qMax = 2.80426233436266

    static let samples: [Double] = [
        0.750993, 0.754683, 0.760006, 0.768128, 0.780215, 0.797434, 0.820951, 0.851932, 0.891543, 0.940952,
        1.001324, 1.073414, 1.156328, 1.248762, 1.349410, 1.456967, 1.570129, 1.687589, 1.808044, 1.930187,
        2.052714, 2.174109, 2.292018, 2.403874, 2.507114, 2.599170, 2.677478, 2.739473, 2.782590, 2.804262,
        2.801926, 2.773248, 2.719402, 2.644253, 2.551740, 2.445798, 2.330364, 2.209376, 2.086769, 1.966481,
        1.852449, 1.747843, 1.652766, 1.566556, 1.488550, 1.418087, 1.354502, 1.297135, 1.245322, 1.198400,
        1.155708, 1.116582, 1.080359, 1.046378, 1.013976, 0.982490, 0.951258, 0.919758, 0.888041, 0.856295,
        0.824713, 0.793484, 0.762799, 0.732849, 0.703825, 0.675917, 0.649315, 0.624211, 0.600794, 0.579247,
        0.559711, 0.542320, 0.527206, 0.514502, 0.504342, 0.496857, 0.492182, 0.490448, 0.491789, 0.496338,
        0.504221, 0.515309, 0.529144, 0.545243, 0.563125, 0.582309, 0.602313, 0.622656, 0.642856, 0.662431,
        0.680900, 0.697873, 0.713320, 0.727305, 0.739891, 0.751139, 0.761114, 0.769876, 0.777489, 0.784016,
        0.789519, 0.794061, 0.797704, 0.800512, 0.802546, 0.803869, 0.804545, 0.804635, 0.804202, 0.803309,
        0.802019, 0.800393, 0.798496, 0.796389, 0.794135, 0.791797, 0.789437, 0.787118, 0.784902, 0.782853,
        0.781033, 0.779491, 0.778227, 0.777228, 0.776479, 0.775966, 0.775677, 0.775598, 0.775714, 0.776012,
        0.776479, 0.777101, 0.777864, 0.778754, 0.779758, 0.780863, 0.782054, 0.783318, 0.784641, 0.786009,
        0.787409, 0.788828, 0.790251, 0.791665, 0.793056, 0.794410, 0.795715, 0.796955, 0.798119, 0.799191,
        0.800158, 0.801007, 0.801723, 0.802294, 0.802705, 0.802943, 0.802995, 0.802846, 0.802483, 0.801892,
        0.801059, 0.799977, 0.798657, 0.797119, 0.795379, 0.793457, 0.791370, 0.789135, 0.786772, 0.784298,
        0.781731, 0.779089, 0.776390, 0.773653, 0.770895, 0.768134, 0.765389, 0.762677, 0.760017, 0.757426,
        0.754922, 0.752524, 0.750250, 0.748117, 0.746143, 0.744348, 0.742748, 0.741362, 0.740207, 0.739302,
        0.738665, 0.738314, 0.738267, 0.738542, 0.739157, 0.740129, 0.741478, 0.743220, 0.745375, 0.747960,
        0.750993,
    ]

    /// Normalized shape value at a fractional cycle position (wraps
    /// periodically outside [0,1)), linearly interpolated between the
    /// precomputed samples.
    static func shape(atFraction frac: Double) -> Double {
        var f = frac.truncatingRemainder(dividingBy: 1.0)
        if f < 0 { f += 1.0 }
        let scaled = f * Double(samples.count - 1)
        let i0 = Int(scaled)
        let i1 = min(i0 + 1, samples.count - 1)
        let t = scaled - Double(i0)
        return samples[i0] * (1 - t) + samples[i1] * t
    }

    /// Instantaneous CCA inlet velocity [cm/s] at fractional cycle
    /// position frac — matches cca_waveform_psv_edv.py's
    /// inlet_velocity_ms() exactly (cycle max rescaled to psvCms, cycle
    /// min rescaled to edvCms).
    static func velocityCms(atFraction frac: Double, psvCms: Double, edvCms: Double) -> Double {
        let q = shape(atFraction: frac)
        let t = (q - qMin) / (qMax - qMin)
        return edvCms + (psvCms - edvCms) * t
    }
}
