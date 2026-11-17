# Depth-Resolved Attenuation Methods

`linumpy.intensity.attenuation` exposes four depth-resolved estimators
(DREs) for the OCT attenuation coefficient. They share the same backbone
(Vermeer 2014, Eq. 17) and differ only in how they handle the unknown
signal beyond the bottom of the data range and the noise floor.

## Estimator family

For an A-line $I[i]$ with axial pixel size $\Delta$ and length $N$:

$$
\hat\mu[i] \;=\; \frac{1}{2\Delta}\,
    \ln\!\left(1 + \frac{I[i]}{\sum_{j=i+1}^{N-1} I[j] + C}\right)
$$

The four methods differ only in how $C$ (the *finite-range* term) is
computed and how the input is preconditioned.

| Method | $C$ | Pre-processing | When to use |
|---|---|---|---|
| Vermeer 2014 | $0$ | Raw input | Reference / lower-bound on $\mu$. Severely overestimates $\mu$ in the bottom 10–20 % of the volume. |
| Smith 2015 | $C \approx I[i_{\max}] / (2\,\hat\mu_E\,\Delta)$ | XY median filter | Legacy linumpy default; $\hat\mu_E$ comes from a log-gradient mean. Kept for reproducing past results. |
| Liu 2019 | $C = I[i_{\max}] / (\exp(2\,\hat\mu_E\,\Delta) - 1)$ | XY median filter, per-A-line LSQ tail fit for $\hat\mu_E$ | Drop-in replacement for Smith. The exact denominator avoids the linearization error and matches the geometric tail integral exactly. |
| **`li`** *(current default)* | Same as Liu, but on the noise-subtracted, SNR-truncated A-line | Noise-floor subtraction + per-A-line truncation when SNR drops below `snr_threshold_db` (default 6 dB) | **Default.** Handles the non-negligible noise floor present in real OCT data. Truncates each A-line individually so the bottom voxels do not pollute the estimate; the tail constant is fitted at the cut. The noise floor is the median of the non-zero voxels outside the tissue mask (empty mosaic padding is left out). |

How `li` differs from the paper it is named after (Li et al. 2020, checked
against the article): the paper crops each A-line where the signal is 5 dB above
a system noise floor measured with the sample arm blocked, does not subtract
that floor from the signal, and solves $\mu[n]$ recursively from a boundary
value fitted on the last 100–200 µm of the cropped A-line (its Eq. 10). `li`
here crops the same way but against the level outside the tissue (agarose after
the focus correction), subtracts that level, and uses Liu's closed form with the
tail fitted on the last voxels of the cropped A-line, which is the same model
written with an explicit tail term. Below the crop the estimate keeps the decay
rate of that tail down to the end of the tissue mask. Both papers, and Vermeer
2014, require the focus (confocal) response to be divided out first; Vermeer
subtracts the noise before that division, the pipeline divides first.

The two improvements over Smith are:

1. The exact denominator $\exp(2\,\hat\mu_E\,\Delta) - 1$ instead of its
   first-order Taylor series $2\,\hat\mu_E\,\Delta$. The two agree to
   within a percent for $2\,\mu_E\,\Delta < 0.1$ but diverge fast at
   the resolutions common in benchtop OCT.
2. A per-A-line least-squares fit for $\hat\mu_E$ (Liu, Li) replacing
   Smith's log-gradient mean.

## Implementation

All four methods share a small set of helpers in
`linumpy.intensity.attenuation`:

- `_median_xy_filter(vol, k)` — pre-denoising
- `_auto_tissue_mask(vol, zshift)` — water/tissue interface detection
- `_tail_at_aline_end(vol, mask, n_tail)` — per-A-line LSQ fit of $\ln I$
  vs depth on the last voxels inside the mask, where the summed range ends
- `_exact_tail_C(i_end, mu_E, dz, valid)` — Liu's $C$ via `np.expm1`; an
  A-line whose fit shows no decay takes the median decay rate of the others
- `_finalize_attenuation(attn, mask, fill_holes)` — NaN/mask cleanup

The Vermeer core (cumsum + log over the full volume) optionally runs
on CuPy via `use_gpu=True`; pass it through Smith / Liu / Li to
accelerate the bottleneck on machines with a CUDA GPU. The CPU path is
unchanged when `use_gpu=False` (the default) or when CuPy is not
available.

The Neubrand 2023 paper (J. Biomed. Opt. 28, 066001) is the reference
for the *exact* form (Eq. 17 above). The widely circulated *linearized*
form

$$
\hat\mu[i] \;\approx\; \frac{I[i]}{2\Delta\,\sum_{j=i+1}^{N-1} I[j]}
$$

(Neubrand Eq. 18) systematically over-estimates $\mu$ by
$\mathcal{O}(\mu^2 \Delta)$ and is *not* used in linumpy.

## A separate model: Faber 2004

`get_attenuation_faber2004` uses a fundamentally different model: a
single-scattering signal modulated by a confocal Lorentzian PSF, fit
per A-line by non-linear least squares to recover
$(z_0, z_R, \mu_t)$. It returns one $\mu_t$ value per A-line rather
than a depth-resolved profile and is included for completeness only.

A historical shape bug (the depth array was sliced as `z[zp::]` while
the data was masked; this raised `ValueError` whenever the mask was
non-contiguous) was fixed in this iteration. The numerical accuracy
of the fit remains sensitive to the hard-coded initial guess
(`p0 = [0.0, 100.0, 0.001]`) and is not validated for clinical use;
adjust the constants in the function body for your setup.

## CLI: `linum_compensate_attenuation_inplace`

```bash
linum-compensate-attenuation-inplace input.ome.zarr output.ome.zarr \
    --method {li,liu,smith,vermeer}      # default: li (Li 2020)
    --strength 0.3                       # 1.0 = textbook formula; <1 attenuates
    --k 10                               # XY median filter (voxels); 0 disables
    --zshift 3                           # voxels under interface to skip
    --min_bias 0.05                      # cap maximum gain at 1/min_bias
```

The Nextflow pipeline forwards the choice through
`params.compensate_attenuation_method` (see
`workflows/reconst_3d/nextflow.config`); set it per-subject in
`<subject>/nextflow.config` if a different method is needed.

### Why `--strength` is needed

Vermeer's single-scattering model overestimates effective attenuation
in scattering brain tissue because the multiple-scattering signal floor
violates the geometric-tail assumption. Empirical sweep on cropped
600 µm sub-22 slices, integrated bias `exp(-2*strength*OD)`:

| `--strength` | Z-direction intensity drop |
|---|---|
| pre-correction | +38.6 % |
| 0.25 | +19.0 % |
| 0.28 | +4.2 % |
| **0.30 (default)** | **−4.2 %** |
| 0.40 | −18.9 % |
| 1.00 (textbook) | −130.9 % |

The default of 0.30 yields an essentially flat depth profile on
sub-22 brain data. Other tissue types may need re-tuning.

### Fitting the strength on each slab (`--calibrate_strength`)

One strength fits one acquisition geometry. The depth-resolved estimate
takes the place where an A-line runs out of signal as the end of the
tissue. In a slab imaged with the focus deep below the surface the signal
falls off earlier than in one imaged with the focus near the surface, the
optical depth comes out several times larger, and the same strength
over-corrects it. White matter is over-corrected most, so the stack shows
a contrast step where the sample height was changed between two slabs.

With `--calibrate_strength` the optical depth keeps its lateral pattern
and only its scale is fitted: the strength is the value at which the
median tissue level has no trend over planes 2 to `--calibrate_planes`
below the surface (the planes that go into the stack). The fit is a
bisection between 0 and `--calibrate_max_strength` (2.0; 1.0 is the published
formula, and with the tail fitted at the crop most slabs land between 0.5 and 1.5); `--strength` is used
for a slab with too little tissue. On one brain whose sample height
changed mid-acquisition the fitted values were 0.03–0.18 for the slabs
with the surface at the top of the scan window and 0.56–0.86 for the
slabs with the surface 200 µm lower; the bright-to-grey ratio differed by
21 % across that joint with a fixed strength of 0.45 and by 4 % with the
fitted one, and the plane-to-plane banding of grey tissue halved.

## References

* Vermeer K. A. *et al.* "Depth-resolved model-based reconstruction of
  attenuation coefficients in optical coherence tomography."
  *Biomed. Opt. Express* **5**, 322–337 (2014).
  <https://doi.org/10.1364/BOE.5.000322>
* Smith G. T. *et al.* "Automated, Depth-Resolved Estimation of the
  Attenuation Coefficient From Optical Coherence Tomography Data."
  *IEEE Trans. Med. Imaging* **34**, 2592–2602 (2015).
  <https://doi.org/10.1109/TMI.2015.2450197>
* Liu J. *et al.* "Optimized depth-resolved estimation to measure
  optical attenuation coefficients from optical coherence tomography
  and its application in cerebral damage determination."
  *J. Biomed. Opt.* **24**, 035002 (2019).
  <https://doi.org/10.1117/1.JBO.24.3.035002>
* Li K. *et al.* "Robust, accurate depth-resolved attenuation
  characterization in optical coherence tomography."
  *Biomed. Opt. Express* **11**, 672–687 (2020).
  <https://doi.org/10.1364/BOE.382493>
* Neubrand L. B., van Leeuwen T. G., Faber D. J. "Accuracy and
  precision of depth-resolved estimation of attenuation coefficients
  in optical coherence tomography." *J. Biomed. Opt.* **28**, 066001
  (2023). <https://doi.org/10.1117/1.JBO.28.6.066001>
* Faber D. J. *et al.* "Quantitative measurement of attenuation
  coefficients of weakly scattering media using optical coherence
  tomography." *Opt. Express* **12**, 4353–4365 (2004).
  <https://doi.org/10.1364/OPEX.12.004353>

## Testing results

The synthetic regression suite (`linumpy/tests/test_attenuation.py`,
7 tests, all passing):

| Test | What it covers |
|---|---|
| `test_recovers_uniform_attenuation_from_clean_exponential` | Vermeer recovers $\mu = 50$ /cm to <1 % in the central 60 % of a clean exponential A-line. |
| `test_does_not_underestimate_by_factor_two` | Regression for the historical bug that included $I[i]$ in the tail sum (halved $\mu$). |
| `test_extended_alias_emits_deprecation_warning` | The legacy `get_extended_attenuation_vermeer2013` name still works and warns. |
| `test_liu2019_recovers_uniform_attenuation` | Liu 2019 recovers $\mu$ to within 2 % across the central 60 %. |
| `test_liu2019_tail_more_accurate_than_vermeer` | Demonstrates Liu's exact-form $C$ reduces the tail blow-up vs $C = 0$. |
| `test_li2020_recovers_attenuation_with_noise_floor` | Li 2020 recovers $\mu = 20$ /cm to within 10 % on a 600-voxel A-line with a $10^{-3}$ noise floor. |
| `test_faber2004_runs_with_noncontiguous_mask` | Regression for the Faber `z[zp::]` shape bug. |

The CLI dispatch suite (`scripts/tests/test_compensate_attenuation_inplace.py`,
8 tests, all passing) parametrizes `--method` over all four estimators
and checks each runs end-to-end on a synthetic decay volume and
reduces the depth drop (the `li` run is allowed to over-correct on
the small 60-voxel synthetic since SNR-based truncation leaves too
few voxels for a stable $\hat\mu_E$ fit at that size).

## Real-data sweep

Cropped 42×1832×988 OCT slice (10 µm/voxel axial), `--strength 0.3`,
single-pass `linum-compensate-attenuation-inplace`. Drop % is the
mean intensity drop between the top 1/8 and bottom 1/8 of the volume —
zero would mean a perfectly flat axial profile.

| Method  | Wall time | Top mean | Bottom mean | Residual drop |
|---------|-----------|----------|-------------|---------------|
| input   | —         | 9.8      | 5.9         | 40.16 %       |
| smith   | 26.0 s    | 10.0     | 7.3         | 27.11 %       |
| vermeer | 19.3 s    | 10.2     | 19.3        | −89.55 %      |
| liu     | 23.0 s    | 10.1     | 9.1         |  9.33 %       |
| **li**  | 23.3 s    | 10.1     | 9.2         |  **8.93 %**   |

Liu and Li reduce the residual axial drop ~3× compared to Smith. The
bare Vermeer estimator over-corrects (negative drop) because it
assumes the signal beyond the volume is zero — the well-known
finite-range bias the other three methods address. On this volume Li
and Liu agree closely; Li's noise-floor handling pays off mostly when
the deep portion of the A-line approaches the detector noise level.

## Default choice and why

The pipeline default is **Li 2020** (`--method li`,
`compensate_attenuation_method = 'li'` in the Nextflow config). The
rationale, in order of importance:

1. **Lowest residual axial drop** in the real-data sweep above
   (8.93 % vs Smith's 27.11 %, a ~3× reduction at the same
   `--strength`).
2. **Robust to the detector noise floor.** Real OCT A-lines do not
   decay to zero — they asymptote to the detector noise (~$10^{-3}$
   of the surface intensity in the test volume). Liu's exact-form
   $C$ already prevents the deep blow-up Vermeer suffers from, but
   it still includes the noise tail in $\hat\mu_E$. Li adds two
   cheap safeguards: subtract an estimate of the noise floor before
   the slope fit, and cut each A-line at the depth where its SNR
   drops below 6 dB. The result is that $\hat\mu_E$ is fit only on
   voxels that actually carry signal.
3. **Marginal cost over Liu.** On the test volume the wall time was
   23.3 s (Li) vs 23.0 s (Liu) — the noise / SNR pre-pass is a
   fraction of the cumulative-sum cost, which dominates either way.

When Li is *not* the right default:

* **Very short A-lines** (a few dozen voxels) — SNR-based truncation
  leaves too few samples for a stable $\hat\mu_E$ fit. Use `liu` or
  `smith` instead. The synthetic 60-voxel CLI test
  (`test_method_dispatch`) explicitly exempts `li` from the strict
  flatness assertion for this reason.