# Pipeline Methods

Algorithmic and mathematical description of the linumpy serial-OCT processing
pipelines, prepared for a methods section. Two Nextflow workflows are covered:

1. **Preprocessing** — `workflows/preproc/preproc_rawtiles.nf`
2. **3-D Reconstruction** — `workflows/reconst_3d/soct_3d_reconst.nf`

## Notation

| Symbol | Meaning |
|--------|---------|
| $I(z, y, x)$ | 3-D volume, axes ordered (Z, Y, X) |
| $\mathrm{AIP}(y, x) = \tfrac{1}{N_z}\sum_z I(z, y, x)$ | average intensity projection along Z |
| $\mathrm{NCC}(a, b) = \tfrac{\sum (a-\bar a)(b-\bar b)}{\sqrt{\sum(a-\bar a)^2\sum(b-\bar b)^2}}$ | zero-lag normalized cross-correlation (Pearson) |
| $\delta = (\delta_y, \delta_x)$ | XY translation (pixels unless noted) |
| $\Delta$ | axial voxel size |
| $r$ | lateral resolution (mm/pixel) |

All file/line citations refer to the linumpy source tree.

---

## 1. Preprocessing Pipeline (`preproc_rawtiles.nf`)

**Input.** Raw OCT tiles in directories named `tile_x{X}_y{Y}_z{Z}/`, each
containing `.bin` frames plus acquisition metadata.

**Output.** Per-slice 3-D mosaic grids `mosaic_grid_3d_z{Z}.ome.zarr`, a
pairwise XY shift file `shifts_xy.csv`, and an optional `slice_config.csv`.

### 1.1 Mosaic grid assembly — `linum_create_mosaic_grid_3d.py`

Each slice's tiles are placed on a **regular, non-overlapping grid** indexed by
their `(mx, my)` motor coordinates extracted from the folder name
(<ref_file file="/Users/frans/code/linumpy/linumpy/mosaic/discovery.py" />).
Tiles are resampled to an isotropic output resolution $r_\text{out}$ (default:
native, i.e. no resampling) and written into a sharded OME-Zarr:

$$
\text{tile\_size}_d = \left\lfloor N_d \cdot r_d \cdot 1000 \,/\, r_\text{out} \right\rfloor,
\qquad
\text{mosaic\_shape} = \bigl[N_z,\; N_\text{tiles,y}\,\text{tile\_size}_y,\; N_\text{tiles,x}\,\text{tile\_size}_x\bigr]
$$

(<ref_snippet file="/Users/frans/code/linumpy/scripts/mosaic/linum_create_mosaic_grid_3d.py" lines="342-348" />).
Resampling uses anti-aliased Gaussian pre-filtering followed by spline zoom
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/gpu/interpolation.py" lines="163-195" />).
No overlap blending is applied at this stage — that is deferred to the
reconstruction pipeline.

### 1.2 Galvo shift correction — `--fix_galvo_shift`

The galvo mirror's return sweep records a few extra A-lines at a different
optical state, producing a dark/bright band. Let $A \in \mathbb{R}^{N_a \times N_b}$
be the per-tile AIP and $\bar A(y) = \mathrm{mean}_x A(y,x)$ the median-filtered
(size 5) 1-D profile. With the return-window width $w = n_\text{extra}$ known
from metadata, the band start $p$ is found by matched gradients
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/geometry/galvo.py" lines="62-101" />):

$$
\mathbf g(y) = |\bar A(y{+}1) - \bar A(y)|,\qquad
p = \arg\max_y \mathbf g(y)\cdot \mathbf g(y+w).
$$

The corrective circular shift is $s = N_a - p - w$. A confidence score
$C = 0.40\,c_b + 0.35\,c_s + 0.25\,c_d \in [0,1]$ combines boundary contrast,
significant-drop ratio, and average drop depth
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/geometry/galvo.py" lines="182-250" />);
the fix is applied only when $C \ge$ `galvo_confidence_threshold` (default 0.6).
For multi-tile slices, up to 5 centre tiles are sampled and the best confidence
is penalised by $\sqrt{n_\text{consistent}/n_\text{sampled}}$.

### 1.3 Camera shift correction — `--fix_camera_shift` (legacy)

For old data with a saturated first A-line, the brightest row of the AIP is
rolled to the edge of the A-scan axis and the saturated corner is filled from
its neighbour (<ref_snippet file="/Users/frans/code/linumpy/linumpy/microscope/oct.py" lines="124-133" />).

### 1.4 XY shift estimation from metadata — `linum_estimate_xy_shift_from_metadata.py`

For each consecutive slice pair $(i, i{+}1)$ the mosaic extents
$(x_\text{min}, x_\text{max}, y_\text{min}, y_\text{max})$ in mm are computed
from motor stage positions (<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/discovery.py" lines="66-147" />).
Two boundary estimates of the inter-slice drift are formed:

$$
\Delta x_\text{min} = x_\text{min}^{(i)} - x_\text{min}^{(i+1)},\qquad
\Delta x_\text{max} = x_\text{max}^{(i)} - x_\text{max}^{(i+1)},
$$

and similarly for $y$. The estimate with the **smaller absolute magnitude** is
selected, since grid expansion moves only one boundary while the opposite
boundary reflects true tissue drift
(<ref_snippet file="/Users/frans/code/linumpy/scripts/stacking/linum_estimate_xy_shift_from_metadata.py" lines="94-115" />).
A transition is flagged `reliable=0` when the grid shape changed between slices.
Shifts are stored in mm and pixels in `shifts_xy.csv`.

### 1.5 Slice config generation — `linum_generate_slice_config.py`

Produces `slice_config.csv` with columns `slice_id, use, exclude_reason, ...`.
The first `exclude_first_slices` (default 1) calibration slices are marked
`use=false`. When `--detect_galvo` is set, per-slice galvo confidence is
stamped via `detect_galvo_for_slice`
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/geometry/galvo.py" lines="104-179" />).

### 1.6 Optional QC outputs

- **AIPs** (`generate_aip`): mean Z-projection saved as PNG.
- **Mosaic previews** (`generate_mosaic_preview`): orthogonal-view screenshots.

---

## 2. 3-D Reconstruction Pipeline (`soct_3d_reconst.nf`)

**Input.** Mosaic grids + `shifts_xy.csv` (+ optional `slice_config.csv`).

**Output.** Multi-resolution OME-Zarr volume `{subject}.ome.zarr` with pyramid
levels at 10, 25, 50, 100 µm, plus previews, metrics, and (optionally) an
atlas-aligned volume.

### Stage 1 — Per-slice preprocessing

#### 1a. Resampling (optional, `resolution > 0`)

Mosaic grids are resampled to the target lateral resolution (default 10 µm/px)
via anti-aliased Gaussian + spline zoom
(<ref_file file="/Users/frans/code/linumpy/scripts/mosaic/linum_resample_mosaic_grid.py" />).

#### 1b. Focal curvature correction (optional, `fix_curvature_enabled`)

The tissue–water interface is detected as the strongest smoothed Z-gradient:
$z_0(y,x) = \arg\max_z \partial_z I$ after Gaussian smoothing
($\sigma_{xy}, \sigma_z$) and optional log transform
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/geometry/interface.py" lines="111-188" />).
A smooth multiplicative "flatfield" $F(y,x)$ is fitted to the per-tile interface
depth maps using BaSiC, and the per-A-line correction is

$$
\text{corr}(y,x) = (F(y,x) - 1)\,\bar z_0,\qquad
I'(z, y, x) = I\bigl((z - \text{corr}(y,x)) \bmod N_z,\; y,\; x\bigr)
$$

implemented as a sub-voxel circular roll with linear interpolation between
adjacent integer shifts
(<ref_snippet file="/Users/frans/code/linumpy/scripts/analysis/linum_detect_focal_curvature.py" lines="54-67" />).
A quadratic surface $\hat z_0 = a + by + cx + dy^2 + exy + fx^2$ is also
available in `linumpy/geometry/interface.py:fit_interface` but is not used by
the current script.

#### 1c. Illumination correction (optional, `fix_illum_enabled`)

**Model (BaSiC).** Each tile $A_i$ is decomposed as
$A_i = F \odot B_i + D + E_i$, where $F$ is a smooth multiplicative flat-field,
$D$ an additive dark-field, $B_i$ a per-tile baseline, and $E_i$ a sparse
residual. The optimisation is a reweighted $\ell_1$ problem
(<ref_file file="/Users/frans/code/linumpy/scripts/illumination/linum_fix_illumination_3d.py" />):

$$
\min_{F, D, B_i}\; \sum_i \|A_i - F\odot B_i - D\|_{1,w} + \lambda_F\|F\|_* + \lambda_D\|D\|_1
$$

Correction: $\hat A_i = (A_i - D)/F$. Two backends are available:

- **BaSiCPy** (`fix_illum_backend='basicpy'`): the published solver
  (Peng et al. 2017), PyTorch GPU backend.
- **linum-basic** (`fix_illum_backend='linum-basic'`, default): an in-tree
  reimplementation with auto-tuned regularisation
  $\lambda_s = \mathrm{dct\_energy}(\bar D)/800$,
  $\lambda_d = \mathrm{dct\_energy}(\bar D)/2000$, solved by inexact ALM with
  DCT-domain soft-thresholding
  (<ref_file file="/Users/frans/code/linumpy/scripts/illumination/linum_fix_illumination_basic.py" />).

When `fix_illum_per_z_fit=true`, a separate flat-field is fitted per axial
plane; otherwise one global field is applied to all planes. The dark-field is
optionally estimated externally as a per-pixel low percentile of the pooled
tile stack (default `darkfield_percentile=10`).

### Stage 2 — XY tile stitching

#### 2a. Global transform estimation (optional, `stitch_global_transform`)

A single $2\times2$ affine $M$ is fitted across a pool of mid-brain slices to
capture the instrument's scan-to-stage geometry (slice-invariant). For each
neighbour tile pair, the sub-pixel displacement is measured by **phase
correlation** on the Z-projected overlap
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/registration/phase_correlation.py" lines="44-68" />):

$$
R(\boldsymbol\xi) = \mathcal F^{-1}\!\left\{\frac{\mathcal F(a)\,\overline{\mathcal F(b)}}{|\mathcal F(a)\,\overline{\mathcal F(b)}|}\right\},\qquad
\hat{\boldsymbol\delta} = \arg\max_{\boldsymbol\xi} R(\boldsymbol\xi).
$$

All pair displacements are pooled and a least-squares affine is solved
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/motor.py" lines="333-347" />):

$$
\begin{bmatrix} \vdots \\ \delta_y^{(i)} \\ \delta_x^{(i)} \\ \vdots \end{bmatrix}
= \begin{bmatrix} \vdots \\ r_i\; c_i\; 0\; 0 \\ 0\; 0\; r_i\; c_i \\ \vdots \end{bmatrix}
\begin{bmatrix} a\\b\\c\\d \end{bmatrix},\qquad
M = \begin{bmatrix} a&b\\c&d \end{bmatrix}.
$$

Optional histogram matching and Otsu-based empty-overlap filtering are applied
before phase correlation. The fitted $M$ is reused for every slice.

#### 2b. 3-D stitching with blend-shift refinement — `stitch_3d_with_refinement`

Tile positions are computed from $M$ (or the per-slice LS fit) via
$\mathbf p_{ij} = M\,[i,\, j]^\top$
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/motor.py" lines="545-601" />).
The Lefebvre (2017) motor model extracts scan-to-stage rotation
$\theta = \arctan2(-b, d)$, motor-axis non-perpendicularity
$\phi = \arctan2(a, c) + \theta$, and effective overlaps
$O_x = 1 - \sqrt{b^2+d^2}/W$, $O_y = 1 - \sqrt{a^2+c^2}/H$.

**Blend-shift refinement** (`refinement_mode=blend_shift`): for each tile, the
mean sub-pixel offset from all neighbour phase-correlation estimates is
applied as a cubic-interpolation shift to the tile content before blending
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/motor.py" lines="645-681" />):

$$
\bar{\delta} = \frac{1}{2n}\sum_{k=1}^{n} \boldsymbol\delta_k,\qquad
\text{tile}' = \mathrm{shift}(\text{tile},\; (0, \bar\delta_y, \bar\delta_x),\;\text{order}=1).
$$

Shifts exceeding `max_blend_refinement_px` (default 10) are clamped.

**Blending methods** (`stitch_blending_method`, default `diffusion`):

- `none`: overwrite.
- `average`: $\alpha = 0.5$ in overlap, $\alpha = 1$ elsewhere
  (<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/grid.py" lines="686-701" />).
- `diffusion`: solve Laplace's equation on the overlap mask with tile
  boundaries as Dirichlet conditions
  (<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/grid.py" lines="734-752" />):
  $$\nabla^2 u = 0 \;\text{in}\; \Omega,\quad u|_{\partial\Omega_1}=t_1,\; u|_{\partial\Omega_2}=t_2.$$

### Stage 3 — Per-slice corrections

#### 3a. Beam-profile / PSF correction (`compensate_psf_enabled`)

Two estimators:

- **Model-free** (default, `compensate_psf_method='model_free'`): the
  tissue-averaged axial profile $p(z) = \mathrm{median}_{(y,x)\in\mathcal M} I(z,y,x)$
  is estimated from agarose (Otsu-masked) A-lines, background-subtracted and
  normalised, then divided out
  (<ref_file file="/Users/frans/code/linumpy/scripts/attenuation/linum_compensate_psf_model_free.py" />):
  $$I'(z,y,x) = \frac{I(z,y,x)}{1 + \mathrm{psf}(z)},\qquad \mathrm{psf}(z) = \frac{p(z) - p_\text{bg}}{p_\text{bg}}.$$

- **Confocal model** (`compensate_psf_method='model'`): a confocal Lorentzian
  PSF $\mathrm{PSF}(z; z_f, z_R, A) = A/[(z-z_f)^2/z_R^2 + 1]$ is fit to agarose
  profiles by grid search + iterative refinement
  (<ref_file file="/Users/frans/code/linumpy/linumpy/psf/extract.py" />), then
  $$I' = (I - p_\text{bg})/\mathrm{PSF}_{3D} + p_\text{bg}.$$
  The model fit falls back to model-free on failure.

#### 3b. Interface cropping — `crop_interface`

The tissue surface is detected as $z_0 = \arg\max_z \partial_z I$ (smoothed
Gaussian, restricted to the top half of Z) and the volume is cropped to
$I[z_0 : z_0 + D]$ with $D = $ `crop_interface_out_depth` (default 600 µm)
(<ref_file file="/Users/frans/code/linumpy/scripts/mosaic/linum_crop_3d_mosaic_below_interface.py" />).

#### 3c. Depth attenuation compensation (`compensate_attenuation_enabled`)

All four estimators share the Vermeer (2014) depth-resolved attenuation
coefficient (<ref_file file="/Users/frans/code/linumpy/linumpy/intensity/attenuation.py" />):

$$
\hat\mu[i] = \frac{1}{2\Delta}\ln\!\left(1 + \frac{I[i]}{\sum_{j=i+1}^{N-1} I[j] + C}\right).
$$

The methods differ in the finite-range term $C$ and preprocessing:

| Method | $C$ | Preprocessing |
|--------|-----|---------------|
| Vermeer 2014 | $0$ | raw |
| Smith 2015 | $I[i_\text{max}]/(2\hat\mu_E\Delta)$ (linearised) | XY median, log-gradient $\hat\mu_E$ |
| Liu 2019 | $I[i_\text{max}]/(\exp(2\hat\mu_E\Delta)-1)$ (exact) | XY median, per-A-line LSQ $\hat\mu_E$ |
| **Li 2020** (default) | same as Liu | + noise-floor subtraction + SNR truncation at 6 dB |

The attenuation map is converted to a multiplicative bias field via cumulative
optical depth
(<ref_snippet file="/Users/frans/code/linumpy/scripts/attenuation/linum_compensate_attenuation_inplace.py" lines="188-193" />):

$$
\mathrm{OD}[i] = \int_0^i \mu_\text{vox}(z)\,dz,\qquad
\mathrm{bias}[i] = \exp(-2\cdot s\cdot\mathrm{OD}[i]),\qquad
I' = I / \mathrm{bias},
$$

with strength $s = $ `compensate_attenuation_strength` (default 0.3) and
$\mathrm{bias}$ clamped to $\ge$ `min_bias` (default 0.05) to limit noise gain.

#### 3d. Per-slice intensity normalization — `normalize`

For each slice $k$: clip at $P_{99.9}$, subtract the agarose median background,
and divide by a global divisor $P_{90}(P_{\max,z} - \text{bg}_z)$
(<ref_file file="/Users/frans/code/linumpy/scripts/illumination/linum_normalize_intensities_per_slice.py" />):

$$
I'_k = \frac{\min(I_k,\, P_{99.9}(I_k)) - \mathrm{median}(I_k[\text{agarose}])}{P_{90}(\{P_{\max,z} - \text{bg}_z\}_k)}.
$$

### Stage 3.5 — Auto slice quality assessment (optional, `auto_assess_quality`)

GPU-accelerated quality scoring on centre-cropped ROIs; slices below
`auto_assess_min_quality` (default 0.3) are stamped `auto_excluded` in
`slice_config.csv`, merged with any manual exclusions
(<ref_file file="/Users/frans/code/linumpy/scripts/analysis/linum_assess_slice_quality.py" />).

### Stage 4 — Common-space alignment

#### 4a. Re-homing detection (optional, `detect_rehoming`)

For each pairwise step $s_i$ with $\|s_i\| >$ `rehoming_max_shift_mm`, a glitch
is detected when an adjacent step nearly cancels it
(<ref_file file="/Users/frans/code/linumpy/scripts/analysis/linum_detect_rehoming.py" />):

$$
\exists\, j\in\{i{-}1,\, i{+}1\}:\; \|s_i + s_j\| < f_\text{return}\,\|s_i\|,
\qquad f_\text{return} = \texttt{rehoming\_return\_fraction}\;(0.4).
$$

Glitches are replaced by the local median of non-outlier neighbours; genuine
re-homing events (large steps that persist) are preserved. Output:
`shifts_xy_clean.csv` with a `reliable` column.

#### 4b. Bring to common space — `bring_to_common_space`

Pairwise shifts are integrated into cumulative positions and centred on the
middle slice (<ref_file file="/Users/frans/code/linumpy/linumpy/stack_alignment/io.py" />):

$$
C_k = \sum_{i\le k} s_i\;[\text{mm}],\qquad
P_k = C_k / r\;[\text{px}],\qquad
P'_k = P_k - P_{k_\text{mid}}.
$$

Excluded-slice shifts are replaced by `local_median` of neighbours (default).
When `common_space_refine_unreliable=true`, transitions flagged `reliable=0`
are re-estimated by 2-D phase cross-correlation on the central-20% Z
max-projection, gated by `refine_min_correlation` and
`refine_max_discrepancy_px`. Each slice is resampled onto the common canvas via
a SimpleITK `TranslationTransform` with linear interpolation
(<ref_file file="/Users/frans/code/linumpy/linumpy/imaging/transform.py" />).

### Stage 5 — Missing-slice interpolation (optional, `interpolate_missing_slices`)

**zmorph** (default): the boundary planes of the two neighbouring slices are
registered with a 2-D affine $T = (M, \mathbf t)$. For each output plane at
fractional depth $\alpha \in [0,1]$
(<ref_file file="/Users/frans/code/linumpy/linumpy/mosaic/interpolation.py" />):

$$
P_\alpha(\mathbf x) = (1-\alpha)\,B_\text{before}(T^{\alpha}\mathbf x) + \alpha\,B_\text{after}(T^{\alpha-1}\mathbf x),
$$

where the fractional affine $T^\alpha$ uses the matrix fractional power
$M^\alpha$ (Schur decomposition) and translation
$\mathbf t_\alpha = (I - M^\alpha)(I - M)^{-1}\mathbf t$ (Log-Euclidean
framework; Arsigny et al. 2006). Blending is Gaussian-feathered via distance
transforms. Quality gates hard-skip (no volume) when
$\mathrm{NCC} < $ `min_overlap_correlation` or no foreground planes exist.
A rejected warp morphs with identity ($T=I$) from the two cut-adjacent
planes rather than blending interiors.

### Stage 6 — Pairwise registration — `register_pairwise`

For each consecutive pair (fixed $F$, moving $M$):

1. **Z-plane matching.** The best-matching fixed Z-plane is found by NCC on a
   centre-cropped 25% margin-free ROI over a search range around the expected
   overlap (<ref_snippet file="/Users/frans/code/linumpy/linumpy/registration/refinement.py" lines="40-66" />):
   $$z^\star = \arg\max_z \mathrm{NCC}\bigl(F[z,\,\text{crop}],\; M[0,\,\text{crop}]\bigr).$$

2. **Intensity refinement** (SimpleITK, <ref_snippet file="/Users/frans/code/linumpy/linumpy/registration/refinement.py" lines="122-142" />):
   $$\hat T = \arg\min_{T\in\mathcal E} -\mathrm{NCC}(F_{z^\star},\; M_0\circ T),$$
   with $\mathcal E$ = Euler2D (rotation + translation, default) or
   Translation2D, optimised by regular-step gradient descent (learning rate
   1.0, 200 iterations, 3 pyramid levels with shrink factors $[4,2,1]$ and
   smoothing sigmas $[2,1,0]$). Optional initial alignment via Sobel gradient
   magnitude phase correlation and/or centre-of-mass offset. Parameters are
   clamped to `registration_max_translation` (200 px) and
   `registration_max_rotation` (5°); a fallback identity transform is written
   on bound-hit or metric failure.

The 2-D transform is embedded in a 3-D Euler transform (rotation about Z,
in-plane translation) and saved as `.tfm`; the Z-match index is stored
separately in `offsets.txt`.

### Stage 6.5 — Manual alignment refinement (optional)

`refine_manual_transforms` re-runs pairwise registration initialised from a
manual transform, searching a tight residual (default 10 px, 2°)
(<ref_file file="/Users/frans/code/linumpy/scripts/stitching/linum_refine_manual_transforms.py" />).

### Stage 6.75 — Auto-exclude clusters (optional, `auto_exclude_enabled`)

Consecutive pairs with $z_\text{correlation} < $ `auto_exclude_z_corr` (0.6)
are accumulated; runs of length $\ge$ `auto_exclude_consecutive` (3) are
stamped `auto_excluded` in `slice_config.csv`
(<ref_snippet file="/Users/frans/code/linumpy/scripts/analysis/linum_auto_exclude_slices.py" lines="97-116" />).
Excluded slices fall back to motor-only positioning during stacking.

### Stage 7 — Volume stacking — `stack`

#### 7a. XY placement

Cumulative motor shifts are computed (as in §4b), converted to pixels, centred
on the middle slice, and the output canvas is sized to bound all slice extents
(<ref_file file="/Users/frans/code/linumpy/linumpy/stack_alignment/motor_stack.py" />).

#### 7b. Z-overlap matching

Three modes (<ref_file file="/Users/frans/code/linumpy/scripts/stacking/linum_stack_slices_motor.py" />):

- **Expected overlap** (default, `use_expected_z_overlap`):
  $o = N_z - \text{moving\_z} - \lfloor\text{slicing\_interval}/\Delta\rfloor$,
  optionally refined by searching $\pm$`blend_z_refine_vox` voxels for the
  best-NCC tissue boundary.
- **Registration-derived**: $o = N_z - \text{fixed\_z}$ from the pairwise
  transform.
- **Correlation-based** (`find_z_overlap`): NCC sweep over
  $[o_\text{exp} - \Delta_\text{search},\; o_\text{exp} + \Delta_\text{search}]$
  on a centre ROI (<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/stacking.py" lines="98-175" />).

Outlier overlaps are corrected by local median interpolation.

#### 7c. Pairwise transform application

Each slice's 2-D Euler transform is applied per Z-plane via SimpleITK
(<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/stacking.py" lines="178-258" />).
Rotation is clamped to `max_rotation_deg` (5°). A confidence score gates
degradation (<ref_snippet file="/Users/frans/code/linumpy/linumpy/metrics/collectors.py" lines="281-290" />):

$$
c = 0.5\,\max(0, z_\text{corr}) + 0.3\,(1 - \|\tau\|/\tau_\text{max}) + 0.2\,(1 - |\theta|/\theta_\text{max}).
$$

Slices with $c \ge$ `transform_confidence_high` (0.6) get the full transform;
$c \in$ [`transform_confidence_low`, `high`] get rotation-only; $c <$ `low`
(0.3) get identity.

#### 7d. Cumulative translation accumulation

When `stack_accumulate_translations=true`, pairwise translations are
accumulated as canvas offsets (with optional confidence weighting), smoothed by
a 1-D Gaussian ($\sigma =$ `stack_translation_smooth_sigma`, 3.0 slices), and
clamped to `stack_max_cumulative_drift_px` (50 px) from the motor baseline
(<ref_file file="/Users/frans/code/linumpy/linumpy/stack_alignment/motor_stack.py" />):

$$
u_k = \mathrm{smooth}_\sigma\!\Bigl(\sum_{i\le k} c_i\,\tau_i\Bigr),\qquad
\|u_k\| \le \text{drift\_max}.
$$

Per-slice rotations are moving-average-smoothed
(`stack_smooth_window`, 5 slices).

#### 7e. Z-blending

In the overlap region a Hann (cosine) ramp cross-fades the fixed and moving
volumes (<ref_snippet file="/Users/frans/code/linumpy/linumpy/mosaic/stacking.py" lines="340-384" />):

$$
w(z) = \tfrac12\bigl(1 - \cos\tfrac{\pi z}{n_z-1}\bigr),\qquad
B = (1-w)\,F + w\,M.
$$

Intensity matching: by default a scalar median scale (clamped to $[0.5, 2]$)
is applied to the incoming slice. When `stack_overlap_z_gain` is enabled, that
scale is skipped. Instead a 1-D gain is fitted on the **Z-end overlap only**
(previous bottom vs next top after `moving_z_start`, both-tissue voxels) as
$\log(I_\text{next}/I_\text{prev}) = a + b i$, then applied to the previous slab
(ramp from 1 at the surface to $e^{a+bi}$ in the overlap). Optional XY
phase-correlation refinement ($\le$ `blend_refinement_px`) is applied before
blending.

#### 7f. Pyramid construction

The output OME-Zarr is finalised with multi-resolution levels at
`pyramid_resolutions` (default $[10, 25, 50, 100]$ µm). Isotropic resampling
scales each axis independently to the target spacing
(<ref_file file="/Users/frans/code/linumpy/linumpy/io/zarr.py" />).

### Stage 8 — Bias-field correction (optional, `correct_bias_field`)

N4 bias correction assumes a multiplicative bias $s(x) = u(x)\,b(x)$; in log
space, $\log s = \log u + \log b + n$. The N4 algorithm alternates histogram
sharpening (Wiener deconvolution of the log-histogram) and cubic B-spline
fitting of $\log b$ across a multi-resolution pyramid
(<ref_file file="/Users/frans/code/linumpy/scripts/illumination/linum_correct_bias_field.py" />).

Pre-passes: per-Z-plane histogram matching to a global reference CDF, and
Gaussian-smoothed Z-profile gain correction. Modes: `per_section`,
`global`, or `two_pass` (default). Strength blend:
$I_\text{out} = s\cdot I_\text{corr} + (1-s)\,I_\text{in}$.

CPU backend: SimpleITK `N4BiasFieldCorrectionImageFilter`. GPU backend: CuPy
port with separable B-spline fitting and FFT Wiener sharpening.

### Stage 9 — Report generation (optional, `generate_report`)

Aggregates per-step metrics (`*_metrics.json`) into an HTML/zip report with
OK/warning/error status flags.

### Stage 10 — Atlas registration (optional, `align_to_ras_enabled`)

Rigid registration of the stacked volume to the Allen Mouse Brain Atlas (CCF)
in RAS orientation (<ref_file file="/Users/frans/code/linumpy/linumpy/reference/allen.py" />):

$$
\hat T = \arg\min_{T\in\mathrm{SE}(3)} -\mathrm{MI}(A,\; V\circ T),
$$

with Mattes mutual information (50 histogram bins), regular-step gradient
descent (learning rate 0.5, 1000 iterations), 4-level pyramid
(shrink $[8,4,2,1]$, physical sigmas $[4,2,1,0]$). Initial alignment by
centre-translation and optional MOMENTS initializer. The atlas is reoriented
from PIR to RAS+ (PermuteAxes + Flip). The output is resampled into the atlas
grid at all pyramid levels, and the 4×4 affine is stored in OME-Zarr
`.zattrs`.

---

## 3. Key references

| Method | Reference |
|--------|-----------|
| BaSiC illumination correction | Peng et al., *Nat. Commun.* 8, 14836 (2017) |
| Phase-correlation stitching | Preibisch et al., *BMC Bioinformatics* 10, 411 (2009) |
| Motor displacement model | Lefebvre et al. (2017) |
| Log-Euclidean fractional affine | Arsigny et al., *MICCAI* 2006 |
| OCT attenuation (Vermeer) | Vermeer et al., *Biomed. Opt. Express* 5, 322 (2014) |
| OCT attenuation (Smith) | Smith et al., *IEEE TMI* 34, 2592 (2015) |
| OCT attenuation (Liu) | Liu et al., *J. Biomed. Opt.* 24, 035002 (2019) |
| OCT attenuation (Li) | Li et al., *Biomed. Opt. Express* 11, 672 (2020) |
| Attenuation accuracy review | Neubrand et al., *J. Biomed. Opt.* 28, 066001 (2023) |
| Confocal PSF model | Faber et al., *Opt. Express* 12, 4353 (2004) |
| N4 bias correction | Tustison et al., *IEEE TMI* 29, 1310 (2010) |
| N3 bias correction | Sled et al., *IEEE TMI* 17, 87 (1998) |
| Allen CCF atlas | Allen Institute for Brain Science, CCF v3 |

---

## 4. Default parameter summary

Key defaults from `workflows/reconst_3d/nextflow.config`:

| Parameter | Default | Stage |
|-----------|---------|-------|
| `resolution` | 10 µm | 1a |
| `fix_illum_enabled` | true | 1c |
| `fix_illum_backend` | linum-basic | 1c |
| `stitch_overlap_fraction` | 0.2 | 2 |
| `stitch_blending_method` | diffusion | 2b |
| `max_blend_refinement_px` | 10 | 2b |
| `compensate_psf_method` | model_free | 3a |
| `crop_interface_out_depth` | 600 µm | 3b |
| `compensate_attenuation_method` | li | 3c |
| `compensate_attenuation_strength` | 0.3 | 3c |
| `clip_percentile_upper` | 99.9 | 3d |
| `detect_rehoming` | true | 4a |
| `rehoming_return_fraction` | 0.4 | 4a |
| `interpolate_missing_slices` | true | 5 |
| `interpolation_method` | zmorph | 5 |
| `registration_transform` | euler | 6 |
| `registration_max_rotation` | 5.0° | 6 |
| `registration_slicing_interval_mm` | 0.200 | 6/7 |
| `registration_allowed_drifting_mm` | 0.100 | 6/7 |
| `auto_exclude_enabled` | true | 6.75 |
| `stack_blend_enabled` | true | 7e |
| `stack_overlap_z_gain` | false | 7e |
| `stack_accumulate_translations` | true | 7d |
| `stack_max_cumulative_drift_px` | 50 | 7d |
| `stack_translation_smooth_sigma` | 3.0 | 7d |
| `transform_confidence_high/low` | 0.6 / 0.3 | 7c |
| `pyramid_resolutions` | [10, 25, 50, 100] µm | 7f |
| `pyramid_make_isotropic` | true | 7f |
| `correct_bias_field` | false | 8 |
| `bias_mode` | two_pass | 8 |
| `align_to_ras_enabled` | false | 10 |
| `allen_metric` | MI | 10 |
