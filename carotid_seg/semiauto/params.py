"""Explicit parameters for semi-automatic CTA lumen segmentation.

None of these is a universal HU threshold for lumen or calcium. Intensity
decisions are fractions of the local contrast estimated around the operator
seeds. Millimetre limits are anatomical stop rules, not a tube model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LumenParams:
    # Seed neighbourhood used only to estimate local lumen intensity.
    # The point must lie in the lumen; it is not a cutting plane.
    seed_radius_mm: float = 1.25
    shell_step_mm: float = 1.0
    max_shell_mm: float = 12.0
    shell_percentile: float = 15.0
    min_shell_voxels: int = 40
    # A drop larger than this between successive shell low-tails is a new
    # compartment (airway), not the tissue around the vessel. The search
    # keeps the last tissue plateau. This is not a lumen HU threshold.
    shell_discontinuity: float = 180.0
    # Below this lumen-to-soft separation the seed model is refused.
    # This is a noise floor, not an acceptance window for the vessel.
    min_separation: float = 25.0

    # (I - soft) / (lumen - soft). Soft tissue fails; partial-volume lumen
    # near this floor is kept and marked uncertain. No closing afterwards.
    membership_min: float = 0.45
    boundary_band: float = 0.10
    # I > lumen + k * (lumen - soft) is brighter than local contrast allows.
    calcium_contrast_k: float = 0.60
    # Still brighter: compact calcium/bone core rather than blooming transition.
    calcium_core_k: float = 1.20
    high_score: float = 0.80
    # Local lumen may follow a modest bolus change. It may not ratchet
    # down into the tissue the front has already leaked into.
    local_lumen_slack: float = 0.15
    context_cell_mm: float = 5.0
    min_local_samples: int = 12
    # Runaway guard if growth enters a large contrast-filled structure.
    # Not an assumed carotid volume.
    max_volume_mm3: float = 30000.0

    # Parent-vessel stop: local centerline radius after the mask exists.
    # Bulb stays because it is well below this multiple of tubular CCA radius.
    max_radius_factor: float = 2.5
    radius_sample_mm: float = 6.0

    # Canal entrance: bone-like ring outside the local lumen, sustained along
    # the distal ICA. Not a plane through the distal seed.
    bone_ring_gap_mm: float = 0.8
    bone_ring_width_mm: float = 3.0
    bone_fraction: float = 0.45
    encasement_length_mm: float = 2.0

    min_spur_mm: float = 2.0
    min_branch_mm: float = 5.0
    eca_max_length_mm: float = 16.0
    eca_ambiguity_ratio: float = 0.75
    eca_seed_max_distance_mm: float = 3.5

    subocclusion_area_mm2: float = 1.5
    endpoint_ignore_mm: float = 3.0

    # Frangi does not add or remove voxels. When enabled, a low response on
    # an already marginal boundary voxel is only an uncertainty flag.
    use_vesselness: bool = False
    vesselness_min: float = 0.05
