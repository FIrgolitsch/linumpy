"""Depth profile of the beam, measured on the agarose and pooled over a brain.

The model-free beam correction divides every plane of a slab by the depth
profile of the agarose around the tissue. Measured on one slab that profile is
not a stable estimate of the beam: the agarose face is not always at the
tissue face, and the agarose share of a slab ranges from most of it to almost
none. The same tissue then comes out flat in one slab and steep in the next.

Here each slab contributes its agarose profile and the depth of its tissue
face. The profiles are aligned on the tissue face and pooled, so every slab of
a brain is divided by the same number at the same depth below its cut face.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

from linumpy.geometry.interface import find_tissue_interface

MIN_AGAROSE_COLUMNS = 500


@dataclass
class SlabBeamProfile:
    """Agarose depth profile of one slab.

    Attributes
    ----------
    profile : list of float
        Mean agarose intensity of every plane. Empty when the slab holds too
        little agarose to measure one.
    tissue_face : int
        Plane of the tissue surface (median over the tissue columns).
    agarose_face : int
        Plane of the agarose surface (median over the agarose columns).
    n_columns : int
        Number of agarose columns the profile is the mean of.
    coverage : float
        Agarose columns as a fraction of the image.
    """

    profile: list[float]
    tissue_face: int
    agarose_face: int
    n_columns: int
    coverage: float

    @property
    def is_usable(self) -> bool:
        """Whether the slab holds enough agarose for its profile to count."""
        return self.n_columns >= MIN_AGAROSE_COLUMNS and len(self.profile) > 1 and max(self.profile) > 0

    def save(self, path: Path | str) -> None:
        """Write the profile as JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> SlabBeamProfile:
        """Read a profile written by :meth:`save`."""
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class PooledBeamProfile:
    """Beam profile of a brain, by depth from the tissue face.

    Attributes
    ----------
    depth : list of int
        Depth in planes from the tissue face; negative above it.
    divisor : list of float
        Value every plane at that depth is divided by. Its minimum is 1.
    n_slabs : list of int
        Number of slabs that reach each depth.
    n_used : int
        Number of slabs pooled.
    n_excluded : int
        Number of slabs left out for holding too little agarose.
    """

    depth: list[int]
    divisor: list[float]
    n_slabs: list[int]
    n_used: int
    n_excluded: int = 0
    excluded: list[str] = field(default_factory=list)

    def save(self, path: Path | str) -> None:
        """Write the pooled profile as JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> PooledBeamProfile:
        """Read a pooled profile written by :meth:`save`."""
        return cls(**json.loads(Path(path).read_text()))

    def divisor_for_slab(self, tissue_face: int, n_planes: int) -> np.ndarray:
        """Return the divisor of every plane of a slab whose tissue face is at *tissue_face*.

        Planes outside the pooled depth range take the value at its nearest end.
        """
        planes = np.arange(n_planes, dtype=np.float64) - float(tissue_face)
        return np.interp(planes, np.asarray(self.depth, dtype=np.float64), np.asarray(self.divisor, dtype=np.float64))


def slab_beam_profile(vol: np.ndarray, percentile_max: float | None = None, use_gpu: bool = False) -> SlabBeamProfile:
    """Measure the agarose depth profile and the tissue face of a slab.

    Parameters
    ----------
    vol : ndarray
        Slab in ``(Z, Y, X)`` order.
    percentile_max : float, optional
        Clip values above this percentile before the profile is taken, so a
        few saturated voxels do not carry it.
    use_gpu : bool
        Find the interface on the GPU when one is available.

    Returns
    -------
    SlabBeamProfile
        The profile is empty when the slab has no agarose columns.
    """
    vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = vol
    if percentile_max is not None and vol.size:
        clipped = np.clip(vol, None, np.percentile(vol, percentile_max))

    projection = clipped.mean(axis=0)
    valid = projection > 0.0
    if not valid.any() or np.ptp(projection[valid]) == 0:
        return SlabBeamProfile(profile=[], tissue_face=0, agarose_face=0, n_columns=0, coverage=0.0)

    threshold = threshold_otsu(projection[valid])
    tissue = projection >= threshold
    agarose = valid & ~tissue

    # The interface finder takes the depth axis last.
    interface = np.asarray(find_tissue_interface(np.transpose(vol, (2, 1, 0)), use_gpu=use_gpu)).T
    # A column whose interface is at the first plane has no surface in view.
    agarose &= interface > 0

    n_columns = int(agarose.sum())
    tissue_face = int(np.median(interface[tissue])) if tissue.any() else 0
    if n_columns == 0:
        return SlabBeamProfile(profile=[], tissue_face=tissue_face, agarose_face=0, n_columns=0, coverage=0.0)

    profile = clipped[:, agarose].mean(axis=1, dtype=np.float64)
    return SlabBeamProfile(
        profile=[float(v) for v in profile],
        tissue_face=tissue_face,
        agarose_face=int(np.median(interface[agarose])),
        n_columns=n_columns,
        coverage=float(n_columns / agarose.size),
    )


def pool_beam_profiles(
    slabs: Sequence[SlabBeamProfile],
    min_slabs: int = 3,
    names: Sequence[str] | None = None,
) -> PooledBeamProfile:
    """Pool the agarose profiles of the slabs of a brain, aligned on the tissue face.

    The pooled profile is built from its change between successive depths:
    the median, over the slabs that reach both depths, of the change of the
    log profile. Summing those changes gives a profile that does not step
    where slabs with a deep tissue face run out of planes, which a median of
    the profiles themselves does.

    Parameters
    ----------
    slabs : sequence of SlabBeamProfile
        One profile per slab.
    min_slabs : int
        Fewest slabs that must reach a depth for the change there to count.
        Outside those depths the profile is held constant.
    names : sequence of str, optional
        Name of each slab, used to list the ones left out.

    Returns
    -------
    PooledBeamProfile

    Raises
    ------
    ValueError
        When fewer than *min_slabs* slabs can be pooled.
    """
    names = list(names) if names is not None else [str(i) for i in range(len(slabs))]
    used: list[SlabBeamProfile] = []
    excluded: list[str] = []
    for name, slab in zip(names, slabs, strict=True):
        if slab.is_usable:
            used.append(slab)
        else:
            excluded.append(f"{name}: too little agarose")
    if len(used) < min_slabs:
        raise ValueError(
            f"Only {len(used)} of {len(slabs)} slabs can be pooled, {min_slabs} are needed: {'; '.join(excluded)}"
        )

    first = min(-slab.tissue_face for slab in used)
    last = max(len(slab.profile) - 1 - slab.tissue_face for slab in used)
    depth = np.arange(first, last + 1)
    log_profiles = np.full((len(used), depth.size), np.nan)
    for row, slab in enumerate(used):
        profile = np.asarray(slab.profile, dtype=np.float64)
        floor = profile[profile > 0].min()
        start = -slab.tissue_face - first
        log_profiles[row, start : start + profile.size] = np.log(np.maximum(profile, floor))

    n_slabs = np.sum(~np.isnan(log_profiles), axis=0)
    changes = np.diff(log_profiles, axis=1)
    n_changes = np.sum(~np.isnan(changes), axis=0)
    needed = max(min_slabs, int(np.ceil(0.25 * len(used))))
    pooled_changes = np.zeros(changes.shape[1])
    supported = n_changes >= needed
    if supported.any():
        pooled_changes[supported] = np.nanmedian(changes[:, supported], axis=0)

    log_pooled = np.concatenate([[0.0], np.cumsum(pooled_changes)])
    divisor = np.exp(log_pooled - log_pooled.min())
    return PooledBeamProfile(
        depth=[int(d) for d in depth],
        divisor=[float(v) for v in divisor],
        n_slabs=[int(n) for n in n_slabs],
        n_used=len(used),
        n_excluded=len(excluded),
        excluded=excluded,
    )
