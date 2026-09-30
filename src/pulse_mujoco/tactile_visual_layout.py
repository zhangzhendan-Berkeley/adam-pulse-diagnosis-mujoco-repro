from __future__ import annotations


# The small maps follow the alpha-cropped right-hand artwork. Their fingertip
# tops differ anatomically, while the enlarged comparison maps share one
# baseline so the same local taxel row is directly comparable.
CUSTOM_HAND_PLACEMENTS = {
    "little": (0.045, 0.078),
    "ring": (0.190, 0.033),
    "middle": (0.328, 0.002),
    "index": (0.478, 0.034),
}

CUSTOM_HAND_LARGE_RECTS = {
    "ring": (410, 190, 120, 168),
    "middle": (585, 190, 120, 168),
    "index": (760, 190, 120, 168),
}

# The supplier longitudinal coordinate is already oriented distal (0 mm) to
# proximal (25 mm); it must not be vertically flipped in the viewer.  A
# Finger-length compensation is purely anatomical placement on the artwork;
# it must not be tuned to hide a spatially off-centre physical sample. Keep
# the previously reviewed 3 mm middle-finger offset. Whether the pulse lands
# near each pad centre is enforced by the observable tactile quality gate.
CUSTOM_HAND_LONGITUDINAL_COMPENSATION_MM = {
    "ring": 0.0,
    "middle": 3.0,
    "index": 0.0,
}
