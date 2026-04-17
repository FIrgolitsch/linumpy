#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * 3D RECONSTRUCTION PIPELINE FOR SERIAL OCT DATA
 *
 * Input:  Directory containing mosaic_grid*.ome.zarr files + shifts_xy.csv
 * Output: 3D OME-Zarr volume with multi-resolution pyramid
 */

// -----------------------------------------------------------------------------
// Utility Processes
// -----------------------------------------------------------------------------

process README {
    publishDir "${params.output}/${task.process}", mode: 'move'

    output:
    path "readme.txt"

    script:
    """
    echo "3D reconstruction pipeline" >> readme.txt
    echo "" >> readme.txt
    echo "[Params]" >> readme.txt
    for p in ${params}; do echo " \$p" >> readme.txt; done
    echo "" >> readme.txt
    echo "[Command-line]" >> readme.txt
    echo "${workflow.commandLine}" >> readme.txt
    echo "" >> readme.txt
    echo "[Configuration files]" >> readme.txt
    for c in ${workflow.configFiles}; do echo " \$c" >> readme.txt; done
    """
}

process analyze_shifts {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path(shifts_file)

    output:
    path "shifts_analysis/*"

    script:
    """
    linum_analyze_shifts.py ${shifts_file} shifts_analysis \
        --resolution ${params.resolution} \
        --iqr_multiplier ${params.outlier_iqr_multiplier}
    """
}

// -----------------------------------------------------------------------------
// Diagnostic Processes
// -----------------------------------------------------------------------------

process analyze_rotation_drift {
    publishDir "${params.output}/diagnostics/rotation_analysis", mode: 'copy'

    input:
    path("register_pairwise/*")

    output:
    path "rotation_analysis/*"

    script:
    """
    linum_analyze_registration_transforms.py register_pairwise rotation_analysis \
        --resolution ${params.resolution} \
        --rotation_threshold ${params.diagnostic_rotation_threshold}
    """
}

process analyze_tile_dilation {
    publishDir "${params.output}/diagnostics/dilation_analysis/${slice_id}", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid), path(transform_xy)

    output:
    tuple val(slice_id), path("dilation_analysis_${slice_id}.json"), path("dilation_analysis_${slice_id}.png"), path("dilation_analysis_${slice_id}.txt")

    script:
    """
    linum_analyze_tile_dilation.py ${mosaic_grid} ${transform_xy} dilation_analysis \
        --resolution ${params.resolution} \
        --overlap_fraction ${params.motor_only_overlap} \
        --slice_id ${slice_id}

    mv dilation_analysis/dilation_analysis.json dilation_analysis_${slice_id}.json
    mv dilation_analysis/dilation_analysis.png dilation_analysis_${slice_id}.png
    mv dilation_analysis/dilation_analysis.txt dilation_analysis_${slice_id}.txt
    """
}

process aggregate_dilation_analysis {
    publishDir "${params.output}/diagnostics/aggregated_dilation", mode: 'copy'

    input:
    path("dilation_input/*")

    output:
    path "aggregated_dilation_analysis.json", emit: json
    path "per_slice_correction_factors.csv", emit: csv
    path "aggregated_dilation_report.txt", emit: report
    path "aggregated_dilation_analysis.png", emit: plot

    script:
    """
    linum_aggregate_dilation_analysis.py dilation_input . --pattern "*.json"
    """
}

process stitch_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stitch", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    path "slice_z${slice_id}_motor_only.ome.zarr"

    script:
    def blending = params.motor_only_stitch_blending ?: 'diffusion'
    """
    linum_stitch_motor_only.py ${mosaic_grid} "slice_z${slice_id}_motor_only.ome.zarr" \
        --overlap_fraction ${params.motor_only_overlap} \
        --blending_method ${blending}
    """
}

process stitch_refined {
    publishDir "${params.output}/diagnostics/refined_stitch", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    path "slice_z${slice_id}_refined.ome.zarr"
    path "slice_z${slice_id}_refinements.json", optional: true

    script:
    def refinement_out = params.save_refinement_data ? "--output_refinements slice_z${slice_id}_refinements.json" : ""
    """
    linum_stitch_3d_refined.py ${mosaic_grid} "slice_z${slice_id}_refined.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method diffusion \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        ${refinement_out} -f
    """
}

process compare_stitching {
    publishDir "${params.output}/diagnostics/stitch_comparison", mode: 'copy'

    input:
    tuple val(slice_id), path(motor_stitch), path(refined_stitch)

    output:
    path "slice_z${slice_id}_comparison/*"

    script:
    """
    linum_compare_stitching.py ${motor_stitch} ${refined_stitch} \
        "slice_z${slice_id}_comparison" \
        --label1 "Motor-only" --label2 "Refined" \
        --tile_step ${params.comparison_tile_step}
    """
}

process stack_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stack", mode: 'copy'

    input:
    path("slices/*")
    path(shifts_file)

    output:
    path "motor_only_stack.ome.zarr"
    path "motor_only_stack_preview.png", optional: true

    script:
    def blending_arg = params.motor_only_stack_blending ?: 'none'
    def preview_arg = "--preview motor_only_stack_preview.png"
    """
    linum_stack_motor_only.py slices ${shifts_file} motor_only_stack.ome.zarr \
        --blending ${blending_arg} \
        ${preview_arg}
    """
}

process run_full_diagnostics {
    publishDir "${params.output}/diagnostics", mode: 'copy'

    input:
    path(pipeline_output)

    output:
    path "full_diagnostics/*"

    script:
    """
    linum_diagnose_reconstruction.py ${pipeline_output} full_diagnostics \
        --resolution ${params.resolution} \
        --rotation_threshold ${params.diagnostic_rotation_threshold}
    """
}

process analyze_acquisition_rotation {
    publishDir "${params.output}/diagnostics/acquisition_rotation", mode: 'copy'

    input:
    path(shifts_file)
    path("register_pairwise/*")

    output:
    path "acquisition_rotation_analysis/*"

    script:
    """
    linum_analyze_acquisition_rotation.py ${shifts_file} acquisition_rotation_analysis \
        --registration_dir register_pairwise \
        --resolution ${params.resolution}
    """
}

process generate_report {
    publishDir "$params.output", mode: 'copy'

    input:
    tuple path(zarr), path(zip), path(png), path(annotated_png)
    val subject_name

    output:
    path "${subject_name}_quality_report.${params.report_format ?: 'html'}"

    script:
    def fmt          = params.report_format ?: 'html'
    def verbose_flag = params.report_verbose ? "--verbose" : ""
    def overview_arg = png          ? "--overview_png ${png}"          : ""
    def annotated_arg = annotated_png ? "--annotated_png ${annotated_png}" : ""
    """
    linum_generate_pipeline_report.py ${params.output} ${subject_name}_quality_report.${fmt} \
        --title "Quality Report: ${subject_name}" \
        --format ${fmt} ${verbose_flag} ${overview_arg} ${annotated_arg}
    """
}

// -----------------------------------------------------------------------------
// Preprocessing Processes
// -----------------------------------------------------------------------------

process resample_mosaic_grid {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_resampled.ome.zarr")

    script:
    def script_name = params.use_gpu ? "linum_resample_mosaic_grid_gpu.py" : "linum_resample_mosaic_grid.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    """
    ${script_name} ${mosaic_grid} "mosaic_grid_z${slice_id}_resampled.ome.zarr" \
        -r ${params.resolution} ${gpu_flag} -v
    """
}

process fix_focal_curvature {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_focal_fix.ome.zarr")

    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} "mosaic_grid_z${slice_id}_focal_fix.ome.zarr"
    """
}

process fix_illumination {
    cpus params.processes

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")

    script:
    def script_name = params.use_gpu ? "linum_fix_illumination_3d_gpu.py" : "linum_fix_illumination_3d.py"
    """
    ${script_name} ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" \
        --n_processes ${params.processes} \
        --percentile_max ${params.clip_percentile_upper}
    """
}

// -----------------------------------------------------------------------------
// Stitching Processes
// -----------------------------------------------------------------------------

process generate_aip {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_aip.ome.zarr")

    script:
    """
    linum_aip.py ${mosaic_grid} "mosaic_grid_z${slice_id}_aip.ome.zarr"
    """
}

process estimate_xy_transformation {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(aip)

    output:
    tuple val(slice_id), path("z${slice_id}_transform_xy.npy"), emit: transform
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def script_name = params.use_gpu ? "linum_estimate_transform_gpu.py" : "linum_estimate_transform.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    def motor_flag = params.use_motor_positions_for_stitching ? "--use_motor_positions" : ""
    def overlap_arg = "--initial_overlap ${params.stitch_overlap_fraction}"
    """
    ${script_name} ${aip} "z${slice_id}_transform_xy.npy" \
        ${gpu_flag} ${motor_flag} ${overlap_arg}
    """
}

process stitch_3d {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(mosaic_grid), path(transform_xy)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr"), emit: stitched
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} "slice_z${slice_id}_stitch_3d.ome.zarr" \
        --blending_method ${params.stitch_blending_method}
    """
}

process estimate_global_transform {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path("pool_input/*")
    path(slice_config)

    output:
    path("global_affine.npy"), emit: transform
    path("global_affine.json"), optional: true, emit: diagnostics

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""
    def histogram_arg = params.stitch_global_transform_histogram_match ? "--histogram_match" : ""
    def empty_arg = params.stitch_global_transform_max_empty_fraction != null
        ? "--max_empty_fraction ${params.stitch_global_transform_max_empty_fraction}"
        : ""
    def n_samples_arg = (params.stitch_global_transform_n_samples as int) > 0
        ? "--n_samples ${params.stitch_global_transform_n_samples as int}"
        : ""
    def include_arg = params.stitch_global_transform_slices?.trim()
        ? "--include_slice " + params.stitch_global_transform_slices.toString().split('[,\\s]+').join(' ')
        : ""
    def script_name = params.use_gpu ? "linum_estimate_global_transform_gpu.py" : "linum_estimate_global_transform.py"
    """
    ${script_name} pool_input global_affine.npy \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        ${slice_config_arg} \
        ${include_arg} \
        ${histogram_arg} \
        ${empty_arg} \
        ${n_samples_arg} \
        --seed ${params.stitch_global_transform_seed} \
        --diagnostics_json global_affine.json \
        -f
    """
}

process stitch_3d_with_refinement {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(mosaic_grid), path(input_transform)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr"), emit: stitched
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def transform_arg = input_transform.name != 'NO_TRANSFORM' ? "--input_transform ${input_transform}" : ""
    """
    linum_stitch_3d_refined.py ${mosaic_grid} "slice_z${slice_id}_stitch_3d.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method ${params.stitch_blending_method} \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        ${transform_arg} \
        -f
    """
}

process generate_stitch_preview {
    publishDir "${params.output}/previews/stitched_slices", mode: 'copy'

    input:
    tuple val(slice_id), path(stitched_slice)

    output:
    path "slice_z${slice_id}_stitched.png"

    script:
    """
    linum_screenshot_omezarr.py ${stitched_slice} "slice_z${slice_id}_stitched.png" \
        --z_slice 0
    """
}

// -----------------------------------------------------------------------------
// Correction Processes
// -----------------------------------------------------------------------------

process beam_profile_correction {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(slice_3d)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_axial_corr.ome.zarr"), emit: corrected
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process crop_interface {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr"), emit: cropped
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_crop_3d_mosaic_below_interface.py ${image} "slice_z${slice_id}_crop_interface.ome.zarr" \
        --depth ${params.crop_interface_out_depth} \
        --crop_before_interface \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process normalize {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr"), emit: normalized
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def script_name = params.use_gpu ? "linum_normalize_intensities_per_slice_gpu.py" : "linum_normalize_intensities_per_slice.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    """
    ${script_name} ${image} "slice_z${slice_id}_normalize.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper} \
        --min_contrast_fraction ${params.normalize_min_contrast} ${gpu_flag}
    """
}

// -----------------------------------------------------------------------------
// Alignment Processes
// -----------------------------------------------------------------------------

process detect_rehoming_events {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path(shifts_csv), path(slice_config_in)

    output:
    path "shifts_xy_clean.csv",           emit: corrected_shifts
    path "slice_config.csv",              optional: true, emit: slice_config
    path "diagnostics/*",                 optional: true, emit: diagnostics

    script:
    def diag_arg = params.rehoming_diagnostics ? "--diagnostics diagnostics" : ""
    def frac_arg = params.rehoming_return_fraction ? "--return_fraction ${params.rehoming_return_fraction}" : ""
    def tile_fov_arg = params.tile_fov_mm ? "--tile_fov_mm ${params.tile_fov_mm}" : ""
    def tile_tol_arg = (params.tile_fov_mm && params.tile_fov_tolerance != null) ? "--tile_fov_tolerance ${params.tile_fov_tolerance}" : ""
    def max_shift_arg = params.rehoming_max_shift_mm ? "--max_shift_mm ${params.rehoming_max_shift_mm}" : ""
    def sc_args = slice_config_in.name != 'NO_SLICE_CONFIG'
        ? "--slice_config_in ${slice_config_in} --slice_config_out slice_config.csv"
        : ""
    """
    linum_detect_rehoming.py ${shifts_csv} shifts_xy_clean.csv \
        ${frac_arg} ${max_shift_arg} ${tile_fov_arg} ${tile_tol_arg} ${diag_arg} \
        ${sc_args}
    """
}

// Auto-assess slice quality after normalization.
// When an existing slice_config.csv is provided (name != 'NO_SLICE_CONFIG'),
// it is merged so that manually-excluded slices stay excluded regardless of
// their computed quality score. This prevents the auto-assessment from
// silently re-enabling slices that the user or preproc step flagged as bad.
process auto_assess_quality {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("inputs/*"), path(existing_slice_config)

    output:
    path "slice_config.csv", emit: slice_config

    script:
    def update_args = existing_slice_config.name != 'NO_SLICE_CONFIG'
        ? "--update_existing --existing_config ${existing_slice_config}"
        : ""
    """
    linum_assess_slice_quality.py inputs slice_config.csv \\
        --min_quality ${params.auto_assess_min_quality} \\
        --exclude_first ${params.auto_assess_exclude_first} \\
        --roi_size ${params.auto_assess_roi_size} \\
        --processes ${params.processes} \\
        ${update_args} \\
        -f
    """
}

process bring_to_common_space {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("inputs/*"), path("shifts_xy.csv"), path(slice_config)

    output:
    path "*.ome.zarr"

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""

    def excluded_args = params.common_space_excluded_slice_mode ?
        "--excluded_slice_mode ${params.common_space_excluded_slice_mode} --excluded_slice_window ${params.common_space_excluded_slice_window}" : ""

    def refine_arg = params.common_space_refine_unreliable ? "--refine_unreliable" : ""
    def discrepancy_arg = (params.common_space_refine_unreliable && params.common_space_refine_max_discrepancy_px > 0) ?
        "--refine_max_discrepancy_px ${params.common_space_refine_max_discrepancy_px}" : ""
    def min_corr_arg = (params.common_space_refine_unreliable && params.common_space_refine_min_correlation > 0) ?
        "--refine_min_correlation ${params.common_space_refine_min_correlation}" : ""

    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space \
        ${slice_config_arg} ${excluded_args} ${refine_arg} ${discrepancy_arg} ${min_corr_arg}
    mv common_space/* .
    """
}

process generate_common_space_preview {
    publishDir "${params.output}/common_space_previews", mode: 'copy'

    input:
    tuple val(slice_id), path(slice_zarr)

    output:
    path "slice_z${slice_id}_preview.png"

    script:
    """
    linum_screenshot_omezarr.py ${slice_zarr} "slice_z${slice_id}_preview.png"
    """
}

// Interpolate a single missing slice via z-aware morphing (zmorph).
//
// When zmorph's quality gates fail (low boundary NCC, no reliable affine,
// ...) the script exits successfully WITHOUT producing an interpolated
// zarr — fabricating a blended slice would also be made-up data. The
// manifest fragment carries `interpolation_failed=true` and a specific
// `fallback_reason`; downstream `finalise_interpolation` stamps the result
// into `slice_config_final.csv` and the slot remains a genuine gap in the
// stacked volume.
process interpolate_missing_slice {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(missing_slice_id), path(slice_before), path(slice_after)

    output:
    path "slice_z${missing_slice_id}_interpolated.ome.zarr", optional: true, emit: zarr
    path "slice_z${missing_slice_id}_interpolated_preview.png", optional: true, emit: preview
    path "slice_z${missing_slice_id}_interpolated_diagnostics.json", emit: diagnostics
    path "slice_z${missing_slice_id}_manifest.csv", emit: manifest

    script:
    def preview_opt = params.interpolation_preview ? "--preview slice_z${missing_slice_id}_interpolated_preview.png" : ""
    def slab_opt = params.interpolation_reference_slab_size ? "--reference_slab_size ${params.interpolation_reference_slab_size}" : ""
    def fg_opt = params.interpolation_min_foreground_fraction != null ? "--min_foreground_fraction ${params.interpolation_min_foreground_fraction}" : ""
    def ncc_opt = params.interpolation_min_ncc_improvement != null ? "--min_ncc_improvement ${params.interpolation_min_ncc_improvement}" : ""
    """
    linum_interpolate_missing_slice.py ${slice_before} ${slice_after} \
        "slice_z${missing_slice_id}_interpolated.ome.zarr" \
        --method ${params.interpolation_method} \
        --blend_method ${params.interpolation_blend_method} \
        --registration_metric ${params.interpolation_registration_metric} \
        --max_iterations ${params.interpolation_max_iterations} \
        --overlap_search_window ${params.interpolation_overlap_search_window} \
        --min_overlap_correlation ${params.interpolation_min_overlap_correlation} \
        ${slab_opt} \
        ${fg_opt} \
        ${ncc_opt} \
        --slice_id ${missing_slice_id} \
        --diagnostics slice_z${missing_slice_id}_interpolated_diagnostics.json \
        --manifest_entry slice_z${missing_slice_id}_manifest.csv \
        ${preview_opt}
    """
}

// Merge per-slice interpolation manifest fragments into slice_config.csv so
// downstream tooling and the final report see, per slice, whether it was
// interpolated and with what diagnostics. Fragments are staged into a
// `fragments/` directory and consumed via `linum_interpolate_missing_slice.py
// --finalise`.
process finalise_interpolation {
    publishDir "${params.output}", mode: 'copy'

    input:
    tuple path(slice_config), path("fragments/*")

    output:
    path "slice_config_final.csv"

    script:
    """
    linum_interpolate_missing_slice.py --finalise \\
        --slice_config_in ${slice_config} \\
        --slice_config_out slice_config_final.csv \\
        --fragments fragments
    """
}

// -----------------------------------------------------------------------------
// Registration Processes
// -----------------------------------------------------------------------------

process register_pairwise {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path(fixed_vol), path(moving_vol)

    output:
    path "*"

    script:
    def rotation_flag = params.registration_transform == 'translation' ? "--no_rotation" : "--enable_rotation"
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    linum_register_pairwise.py ${fixed_vol} ${moving_vol} \$dirname \
        --slicing_interval_mm ${params.registration_slicing_interval_mm} \
        --search_range_mm ${params.registration_allowed_drifting_mm} \
        --moving_z_index ${params.moving_slice_first_index} \
        --max_rotation_deg ${params.registration_max_rotation} \
        --max_translation_px ${params.registration_max_translation} \
        --initial_alignment ${params.registration_initial_alignment} \
        ${rotation_flag}
    """
}

// Optional: re-register slice pairs that have a manual transform, using the
// manual alignment as initialisation.  Produces a refined transform that
// combines the manual correction with a tight image-based residual correction.
// Only runs when params.refine_manual_transforms = true.
process refine_manual_transforms {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("slices/*"), path("transforms/*")

    output:
    path "*"

    script:
    """
    linum_refine_manual_transforms.py slices transforms . \
        --manual_transforms_dir ${params.manual_transforms_dir} \
        --max_translation_px ${params.refine_max_translation_px} \
        --max_rotation_deg ${params.refine_max_rotation_deg} -f
    """
}

// Auto-exclude extended clusters of consecutive low-quality registrations.
// Reads pairwise_registration_metrics.json from the registration output and
// stamps `auto_excluded` / `auto_exclude_reason` into slice_config.csv. The
// stacking step reads the same slice_config via `--slice_config` and treats
// those slices as motor-only (transforms force-skipped).
process auto_exclude_slices {
    publishDir "$params.output/$task.process", mode: 'copy'

    input:
    tuple path("transforms/*"), path(slice_config_in)

    output:
    path "slice_config.csv", emit: slice_config

    script:
    """
    linum_auto_exclude_slices.py transforms ${slice_config_in} slice_config.csv \
        --consecutive_threshold ${params.auto_exclude_consecutive} \
        --z_corr_threshold ${params.auto_exclude_z_corr}
    """
}

// -----------------------------------------------------------------------------
// Stacking Processes
// -----------------------------------------------------------------------------

// Export lightweight data package for the manual alignment tool.
// Produces AIP images and copies pairwise transforms into a self-contained
// directory that can be downloaded and opened by the manual alignment widget.
process make_manual_align_package {
    publishDir "$params.output/$task.process", mode: 'copy'

    input:
    tuple path("slices/*"), path("transforms/*")

    output:
    path("manual_align_package"), emit: pkg

    script:
    """
    linum_export_manual_align.py slices transforms manual_align_package \
        --level ${params.manual_align_level} \
        --slices_remote_dir ${params.output}/bring_to_common_space
    """
}

// Stacking: assembles common-space slices into a 3D volume using motor positions
// for XY placement, pairwise registration for rotation/translation refinement,
// and correlation or physics-based Z-matching.
// publishDir mode is conditional: 'symlink' when a downstream step will produce
// the final output (preserves work-dir files for -resume); 'move' when this is last.
process stack {
    publishDir "$params.output/$task.process",
        mode: (params.normalize_z_slices || params.align_to_ras_enabled) ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path("slices/*"), path(shifts_file), path("transforms/*"), path(slice_config), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png"), emit: volume
    path("*_metrics.json"), optional: true, emit: metrics
    path("z_matches.csv"), optional: true, emit: z_matches
    path("stacking_decisions.csv"), optional: true, emit: stacking_decisions

    script:
    def options = ""

    // Blending
    if (params.stack_blend_enabled) options += " --blend"
    if (params.blend_refinement_px > 0) options += " --blend_refinement_px ${params.blend_refinement_px}"
    if (params.stack_blend_z_refine_vox > 0) options += " --blend_z_refine_vox ${params.stack_blend_z_refine_vox}"
    if (params.blend_z_refine_min_confidence > 0) options += " --blend_z_refine_min_confidence ${params.blend_z_refine_min_confidence}"

    // Z-matching
    options += " --slicing_interval_mm ${params.registration_slicing_interval_mm}"
    options += " --search_range_mm ${params.registration_allowed_drifting_mm}"
    options += " --moving_z_first_index ${params.moving_slice_first_index}"
    if (params.use_expected_z_overlap) options += " --use_expected_overlap"
    if (params.z_overlap_min_corr > 0) options += " --z_overlap_min_corr ${params.z_overlap_min_corr}"
    if (params.analyze_shifts) options += " --output_z_matches z_matches.csv"
    options += " --output_stacking_decisions stacking_decisions.csv"

    // Pairwise registration refinements
    if (params.apply_pairwise_transforms) {
        options += " --transforms_dir transforms"
        if (params.apply_rotation_only) options += " --rotation_only"
        options += " --max_rotation_deg ${params.max_rotation_deg}"
        if (params.load_transform_min_zcorr > 0) options += " --load_min_zcorr ${params.load_transform_min_zcorr}"
        if (params.load_transform_max_rotation > 0) options += " --load_max_rotation ${params.load_transform_max_rotation}"
        if (params.skip_error_transforms) options += " --skip_error_transforms"
        if (params.skip_warning_transforms) options += " --skip_warning_transforms"
        options += " --confidence_high ${params.transform_confidence_high}"
        options += " --confidence_low ${params.transform_confidence_low}"
    }

    // Slice config: drives per-slice use/auto_excluded → motor-only fallback
    if (slice_config.name != 'NO_SLICE_CONFIG') {
        options += " --slice_config ${slice_config}"
    }

    // Manual alignment overrides.
    // Skip when refine_manual_transforms is active: the refinement step already
    // baked manual corrections into transforms_for_stack, so passing
    // --manual_transforms_dir again would double-apply them.
    if (params.manual_transforms_dir && !params.refine_manual_transforms) {
        options += " --manual_transforms_dir ${params.manual_transforms_dir}"
    }

    // Cumulative translation accumulation
    if (params.stack_accumulate_translations) {
        options += " --accumulate_translations"
        if (params.stack_confidence_weight_translations)
            options += " --confidence_weight_translations"
        if (params.stack_max_cumulative_drift_px > 0)
            options += " --max_cumulative_drift_px ${params.stack_max_cumulative_drift_px}"
        // stack_max_pairwise_translation > 0 filters clamped translations; 0 = keep all.
        // Set to 0 when skip_error_transforms=false to preserve re-homing boundary corrections.
        if (params.stack_max_pairwise_translation > 0)
            options += " --max_pairwise_translation ${params.stack_max_pairwise_translation}"
    }

    if (params.stack_smooth_window > 0) options += " --smooth_window ${params.stack_smooth_window}"
    if (params.stack_translation_smooth_sigma > 0) options += " --translation_smooth_sigma ${params.stack_translation_smooth_sigma}"
    if (params.stack_translation_min_zcorr > 0) options += " --translation_min_zcorr ${params.stack_translation_min_zcorr}"

    // Slices are already in common space; skip redundant XY shifting
    options += " --no_xy_shift"

    options += pyramidArgs()

    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    def orient = params.ras_input_orientation?.trim()?.replace("'", '') ?: ''
    def orientation_arg = orient ? "--orientation ${orient}" : ''
    """
    linum_stack_slices_motor.py slices ${shifts_file} ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} ${show_lines_flag} ${orientation_arg} --crop_to_tissue
    """
}

// Post-stacking Z-direction intensity normalization.
// 'symlink' when align_to_ras follows; 'move' when this is the final output step.
process normalize_z_intensity {
    publishDir "$params.output/$task.process",
        mode: params.align_to_ras_enabled ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path(stacked_zarr), val(subject_name), val(n_slices), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

    script:
    def n_slices_opt = n_slices > 0 ? "--n_serial_slices ${n_slices}" : ""
    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    def orient = params.ras_input_orientation?.trim()?.replace("'", '') ?: ''
    def orientation_arg = orient ? "--orientation ${orient}" : ''
    def znorm_mode_opts = ""
    if (params.znorm_mode == 'histogram') {
        znorm_mode_opts = "--mode histogram --strength ${params.znorm_strength} --tissue_threshold ${params.znorm_tissue_threshold}"
    } else {
        znorm_mode_opts = "--mode percentile --smooth_sigma ${params.znorm_smooth_sigma} --percentile ${params.znorm_percentile} --max_scale ${params.znorm_max_scale} --min_scale ${params.znorm_min_scale} --strength ${params.znorm_strength}"
    }
    def znorm_pyramid_opts = pyramidArgs()
    """
    linum_normalize_z_intensity.py ${stacked_zarr} ${subject_name}.ome.zarr \
        ${n_slices_opt} \
        ${znorm_mode_opts} \
        ${znorm_pyramid_opts}

    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr

    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} ${show_lines_flag} ${orientation_arg} --crop_to_tissue
    """
}

// Atlas registration to Allen Mouse Brain Atlas. Always the final step when enabled.
process align_to_ras {
    publishDir "$params.output/$task.process", mode: 'move', saveAs: { fn ->
        fn.endsWith('.ome.zarr') ? null : fn
    }

    input:
    tuple path(stacked_zarr), path(zarr_zip), path(png), path(annotated_png)
    val subject_name

    output:
    path "${subject_name}_ras.ome.zarr"
    path "${subject_name}_ras.ome.zarr.zip"
    path "${subject_name}_ras_transform.tfm", optional: true
    path "${subject_name}_ras_preview.png", optional: true
    path "${subject_name}_ras_orientation_preview.png", optional: true

    script:
    def orientation_arg = params.ras_input_orientation ? "--input-orientation ${params.ras_input_orientation}" : ""
    def rotation_arg = params.ras_initial_rotation ? "--initial-rotation ${params.ras_initial_rotation}" : ""
    def preview_arg = params.allen_preview ? "--preview ${subject_name}_ras_preview.png" : ""
    def orientation_preview_arg = params.ras_orientation_preview ? "--orientation-preview ${subject_name}_ras_orientation_preview.png" : ""
    def ras_pyramid_opts = pyramidArgs('--n-levels')
    """
    linum_align_to_ras.py ${stacked_zarr} ${subject_name}_ras.ome.zarr \
        --allen-resolution ${params.allen_resolution} \
        --metric ${params.allen_metric} \
        --max-iterations ${params.allen_max_iterations} \
        --level ${params.allen_registration_level} \
        ${orientation_arg} ${rotation_arg} ${preview_arg} ${orientation_preview_arg} \
        ${ras_pyramid_opts}
    zip -r ${subject_name}_ras.ome.zarr.zip ${subject_name}_ras.ome.zarr
    """
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Build pyramid-related CLI arguments from `params.pyramid_*` settings.
// `nLevelsFlag` names the downstream flag (`--n_levels` for most scripts,
// `--n-levels` for `linum_align_to_ras.py`).
def pyramidArgs(nLevelsFlag = '--n_levels') {
    def opts = ""
    if (params.pyramid_n_levels != null) {
        opts += " ${nLevelsFlag} ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid.contains(base_res)) valid = [base_res] + valid
        opts += " --pyramid_resolutions " + valid.collect { it.toString() }.join(' ')
        opts += params.pyramid_make_isotropic ? " --make_isotropic" : " --no_isotropic"
    }
    return opts
}

// Extract z## slice ID string from a filename; returns "unknown" if not found.
def extractSliceId(filename) {
    def name = filename instanceof Path ? filename.getName() : filename.toString()
    def matcher = name =~ /z(\d+)/
    return matcher ? matcher[0][1] : "unknown"
}

// Extract slice ID as integer; returns -1 if not found.
def extractSliceIdInt(filename) {
    def id = extractSliceId(filename)
    return id == "unknown" ? -1 : id.toInteger()
}

// Return tuple(slice_id, file) for a given file path.
def toSliceTuple(file_path) {
    tuple(extractSliceId(file_path), file_path)
}

// Return sorted, comma-separated slice IDs from a list of files (e.g. "01,02,03,05").
def extractSliceIdsString(fileList) {
    fileList
        .collect { extractSliceId(it) }
        .findAll { it != "unknown" }
        .sort { it.toInteger() }
        .join(',')
}

// Remove duplicate and trailing slashes from a path string.
def normalizePath(path) {
    return path.replaceAll('/+', '/').replaceAll('/$', '')
}

// Join path components safely.
def joinPath(base, filename) {
    return "${normalizePath(base)}/${filename}"
}

// Parse a slice_config.csv and return the set of slice IDs marked for use.
def parseSliceConfig(configPath) {
    def slicesToUse = [] as Set
    def slicesExcluded = [] as Set
    def file = new File(configPath)

    if (!file.exists()) error("Slice config file not found: ${configPath}")

    file.withReader { reader ->
        reader.readLine() // Skip header
        reader.eachLine { line ->
            def parts = line.split(',')
            if (parts.size() >= 2) {
                def sliceId = parts[0].trim()
                def use = parts[1].trim().toLowerCase()
                if (use in ['true', '1', 'yes']) slicesToUse.add(sliceId)
                else slicesExcluded.add(sliceId)
            }
        }
    }

    log.info "Slice config: ${slicesToUse.size()} to USE, ${slicesExcluded.size()} EXCLUDED"
    return slicesToUse
}

// Detect single-slice gaps in a sorted slice list.
// Returns a list of [missingId, beforeId, afterId] tuples.
def detectSingleGaps(sliceList) {
    def gaps = []
    def sliceIds = sliceList
        .collect { extractSliceIdInt(it) }
        .findAll { it >= 0 }
        .sort()

    for (int i = 0; i < sliceIds.size() - 1; i++) {
        def current = sliceIds[i]
        def next = sliceIds[i + 1]
        def gap = next - current

        if (gap == 2) {
            def missingId = String.format("%02d", current + 1)
            def beforeId = String.format("%02d", current)
            def afterId = String.format("%02d", next)
            gaps.add([missingId, beforeId, afterId])
            log.info "Gap detected: slice ${missingId} (between ${beforeId} and ${afterId})"
        } else if (gap > 2) {
            log.warn "Multiple missing slices between ${current} and ${next} - cannot interpolate"
        }
    }
    return gaps
}

// Parse debug_slices parameter; supports "25,26", "25-29", or "25,27-29".
// Returns a set of zero-padded slice IDs, or null if not specified.
def parseDebugSlices(debugSlicesStr) {
    if (!debugSlicesStr || debugSlicesStr.trim().isEmpty()) return null

    def sliceIds = [] as Set
    debugSlicesStr.split(',').each { part ->
        part = part.trim()
        if (part.contains('-')) {
            def rangeParts = part.split('-')
            if (rangeParts.size() == 2) {
                def start = rangeParts[0].trim().toInteger()
                def end = rangeParts[1].trim().toInteger()
                (start..end).each { sliceIds.add(String.format("%02d", it)) }
            }
        } else {
            sliceIds.add(String.format("%02d", part.toInteger()))
        }
    }
    return sliceIds
}

// =============================================================================
// MAIN WORKFLOW
// =============================================================================

workflow {
    README()

    def inputDir = normalizePath(params.input)

    // Resolve subject name from path if not explicitly set
    def subject_name = params.subject_name
    if (!subject_name) {
        def pathParts = inputDir.split('/')
        def subMatch = pathParts.find { it ==~ /sub-\w+/ }
        if (subMatch) {
            subject_name = subMatch
        } else {
            def inputFile = file(inputDir)
            def dirName = inputFile.getName()
            subject_name = (dirName in ['mosaic-grids', 'mosaics', 'mosaic_grids', 'input', 'data'])
                ? (inputFile.getParent()?.getName() ?: dirName)
                : dirName
        }
    }
    log.info "Subject: ${subject_name}"
    log.info "GPU: ${params.use_gpu ? 'ENABLED' : 'DISABLED'}"

    def debugSlices = parseDebugSlices(params.debug_slices)
    if (debugSlices) {
        log.info "DEBUG MODE: Processing only slices ${debugSlices.sort().join(', ')}"
    }

    // Shifts file
    def shifts_xy_path = params.shifts_xy ?: "${inputDir}/shifts_xy.csv"
    log.info "Shifts file: ${shifts_xy_path}"

    if (!file(shifts_xy_path).exists()) {
        error """
        Shifts file not found: ${shifts_xy_path}

        Please ensure shifts_xy.csv exists in your input directory,
        or specify the path with --shifts_xy /path/to/shifts_xy.csv
        """
    }
    shifts_xy = channel.of(file(shifts_xy_path))

    // Slice config (optional)
    def slice_config_path = params.slice_config ?: joinPath(inputDir, "slice_config.csv")
    def slicesToUse = null
    if (file(slice_config_path).exists()) {
        slicesToUse = parseSliceConfig(slice_config_path)
        log.info "Slice config: ${slice_config_path}"
    } else if (params.slice_config) {
        error("Slice config file not found: ${slice_config_path}")
    }

    // Discover input mosaic grids
    log.info "Looking for mosaic grids in: ${inputDir}"

    def inputDirFile = file(inputDir)
    def mosaicFiles = inputDirFile.listFiles()
        .findAll { it.isDirectory() && it.name.startsWith('mosaic_grid') && it.name.endsWith('.ome.zarr') && it.name =~ /z\d+/ }
        .sort { it.name }

    if (mosaicFiles.isEmpty()) {
        error("No mosaic grids found in ${inputDir}. Expected: mosaic_grid*_z00.ome.zarr")
    }
    log.info "Found ${mosaicFiles.size()} mosaic grids"

    inputSlices = channel
        .fromList(mosaicFiles)
        .map { toSliceTuple(it) }
        .filter { slice_id, _files ->
            if (debugSlices != null) {
                def included = debugSlices.contains(slice_id)
                if (!included) log.debug "Skipping slice ${slice_id} (not in debug_slices)"
                return included
            }
            if (slicesToUse != null) return slicesToUse.contains(slice_id)
            return true
        }

    def has_slice_config = file(slice_config_path).exists() || params.auto_assess_quality
    slice_config_channel = file(slice_config_path).exists()
        ? channel.fromPath(slice_config_path)
        : channel.of(file('NO_SLICE_CONFIG'))

    if (params.analyze_shifts) {
        analyze_shifts(shifts_xy)
    }

    // Stage 1: Preprocessing
    resampled = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices
    focal_fixed = params.fix_curvature_enabled ? fix_focal_curvature(resampled) : resampled
    illum_fixed = params.fix_illum_enabled ? fix_illumination(focal_fixed) : focal_fixed

    // Stage 2: XY Stitching (image-registration-based blend refinement)
    if (params.stitch_global_transform) {
        pooled_mosaics = illum_fixed.map { _id, p -> p }.collect()
        slice_config_file = file(slice_config_path).exists() ? file(slice_config_path) : file('NO_SLICE_CONFIG')
        estimate_global_transform(pooled_mosaics, slice_config_file)
        global_transform = estimate_global_transform.out.transform
        stitch_inputs = illum_fixed.combine(global_transform)
    } else {
        no_transform = channel.of(file('NO_TRANSFORM'))
        stitch_inputs = illum_fixed.combine(no_transform)
    }
    stitch_3d_with_refinement(stitch_inputs)
    stitched_slices = stitch_3d_with_refinement.out.stitched

    if (params.stitch_preview) {
        generate_stitch_preview(stitched_slices)
    }

    // Stage 3: Corrections
    beam_profile_correction(stitched_slices)
    crop_interface(beam_profile_correction.out.corrected)
    normalize(crop_interface.out.cropped)

    // Stage 3.5: Auto slice quality assessment (optional)
    // Runs after normalization, generates a slice_config.csv that marks
    // degraded slices. When a static slice_config.csv is present it is
    // merged so that manually-excluded slices remain excluded.
    if (params.auto_assess_quality) {
        auto_assess_inputs = normalize.out.normalized
            .map { _id, norm_path -> norm_path }
            .collect()
        existing_slice_config_file = file(slice_config_path).exists()
            ? file(slice_config_path)
            : file('NO_SLICE_CONFIG')
        auto_assess_quality(auto_assess_inputs.combine(channel.of(existing_slice_config_file)))
        effective_slice_config = auto_assess_quality.out.slice_config
    } else {
        effective_slice_config = slice_config_channel
    }

    // Stage 4: Common Space Alignment
    // Optionally correct encoder glitch spikes before alignment. When a
    // real slice_config is available, detect_rehoming also stamps
    // rehomed/rehoming_reliable flags back into it.
    current_slice_config = effective_slice_config.first()
    if (params.detect_rehoming) {
        detect_rehoming_input = shifts_xy.combine(current_slice_config)
        detect_rehoming_events(detect_rehoming_input)
        aligned_shifts = detect_rehoming_events.out.corrected_shifts
        if (has_slice_config) {
            current_slice_config = detect_rehoming_events.out.slice_config.first()
        }
    } else {
        aligned_shifts = shifts_xy
    }

    common_space_input = normalize.out.normalized
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatten()
        .collate(2)
        .map { _meta, filename -> filename }
        .collect()
        .merge(aligned_shifts) { a, b -> tuple(a, b) }
        .merge(current_slice_config) { a, b -> tuple(a[0], a[1], b) }

    bring_to_common_space(common_space_input)

    slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList { a, b -> a.getName() <=> b.getName() }

    if (params.common_space_preview) {
        preview_input = bring_to_common_space.out
            .flatten()
            .map { toSliceTuple(it) }
        generate_common_space_preview(preview_input)
    }

    // Stage 5: Missing Slice Interpolation (optional)
    //
    // Gaps in the slice sequence are driven by slice_config.csv: any slice
    // with use=false (either set manually during preproc or automatically by
    // auto_assess_quality) is filtered out upstream and then appears as a
    // gap here. detectSingleGaps turns each single-slice gap into an
    // interpolation job. The resulting per-slice diagnostics are merged
    // back into slice_config_final.csv so the record of what happened to
    // each slice survives to the final report.
    if (params.interpolate_missing_slices) {
        gaps_channel = slices_common_space
            .map { sliceList -> [detectSingleGaps(sliceList), sliceList] }
            .flatMap { gapsAndSlices ->
                def gaps = gapsAndSlices[0]
                def sliceList = gapsAndSlices[1]
                if (gaps.isEmpty()) return []

                gaps.collect { gap ->
                    def missingId = gap[0], beforeId = gap[1], afterId = gap[2]
                    def sliceBefore = sliceList.find { it.getName().contains("slice_z${beforeId}") }
                    def sliceAfter = sliceList.find { it.getName().contains("slice_z${afterId}") }
                    (sliceBefore && sliceAfter) ? tuple(missingId, sliceBefore, sliceAfter) : null
                }.findAll { it != null }
            }

        interpolate_missing_slice(gaps_channel)

        // Merge per-slice manifest fragments back into slice_config so the
        // final CSV records which slices were interpolated and with what
        // diagnostics. Skipped when no real slice_config.csv is present
        // (nothing to merge into).
        if (has_slice_config) {
            finalise_input = current_slice_config
                .combine(interpolate_missing_slice.out.manifest.collect())
            finalise_interpolation(finalise_input)
            current_slice_config = finalise_interpolation.out.first()
        }

        all_slices = slices_common_space
            .mix(interpolate_missing_slice.out.zarr.collect())
            .flatten()
            .toSortedList { a, b -> a.getName() <=> b.getName() }
    } else {
        all_slices = slices_common_space
    }

    // Stage 6: Pairwise Registration
    log.info "Registering slices pairwise"

    fixed_slices = all_slices
        .map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }
        .flatten()
    moving_slices = all_slices
        .map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }
        .flatten()
    pairs = fixed_slices.merge(moving_slices)

    register_pairwise(pairs)

    // Stage 7: Stacking
    log.info "Stacking slices with registration refinements"

    slices_collected = all_slices.flatten().collect()
    transforms_collected = register_pairwise.out.collect()

    // Stage 6.5: Export Manual Alignment Data (optional)
    if (params.export_manual_align) {
        export_input = slices_collected
            .combine(transforms_collected)
            .map { items ->
                def slices = []
                def transforms = []
                items.each { item ->
                    def name = item.getName()
                    if (name.endsWith('.ome.zarr')) {
                        slices << item
                    } else if (!name.endsWith('.json')) {
                        transforms << item
                    }
                }
                tuple(slices, transforms)
            }
        make_manual_align_package(export_input)
    }

    // Stage 6.75: Optional refinement of manual transforms.
    // Re-runs pairwise registration initialised from the manual transform for
    // each manually-corrected pair; non-manual pairs are copied unchanged.
    // Only active when both refine_manual_transforms and manual_transforms_dir
    // are set.  The refined outputs replace automated transforms for stacking.
    if (params.refine_manual_transforms && params.manual_transforms_dir) {
        log.info "Refining manual transforms from: ${params.manual_transforms_dir}"
        refine_input = slices_collected
            .combine(transforms_collected)
            .map { items ->
                def slices = items.findAll { it.getName().endsWith('.ome.zarr') }
                def transforms = items.findAll { !it.getName().endsWith('.ome.zarr') }
                tuple(slices, transforms)
            }
        refine_manual_transforms(refine_input)
        transforms_for_stack = refine_manual_transforms.out.collect()
    } else {
        transforms_for_stack = transforms_collected
    }

    // Auto-exclude: detect clusters of consecutive low-quality registrations.
    // The script stamps auto_excluded=true/auto_exclude_reason into slice_config,
    // so downstream stack sees it via --slice_config (no separate CSV).
    // Requires a real slice_config to stamp into.
    stack_slice_config = current_slice_config
    if (params.auto_exclude_enabled && has_slice_config) {
        auto_exclude_input = transforms_for_stack.combine(current_slice_config)
        auto_exclude_slices(auto_exclude_input)
        stack_slice_config = auto_exclude_slices.out.slice_config.first()
    }

    stack_input = slices_collected
        .combine(shifts_xy)
        .combine(transforms_for_stack)
        .combine(stack_slice_config)
        .map { items ->
            def slices = []
            def shifts = null
            def transforms = []
            def sc = null

            items.each { item ->
                def name = item.getName()
                if (name == 'NO_SLICE_CONFIG' || name == 'slice_config.csv' || name == 'slice_config_final.csv') {
                    sc = item
                } else if (name.endsWith('.csv')) {
                    shifts = item
                } else if (name.endsWith('.ome.zarr')) {
                    slices << item
                } else if (!name.endsWith('.json')) {
                    transforms << item
                }
            }

            def slice_ids_str = extractSliceIdsString(slices)
            tuple(slices, shifts, transforms, sc, subject_name, slice_ids_str)
        }

    stack(stack_input)
    stack_output = stack.out.volume
    stack_metadata = stack_input.map { slices, shifts, transforms, sc, name, ids_str ->
        tuple(name, ids_str.split(',').size(), ids_str)
    }

    // Stage 8: Z-Intensity Normalization (optional)
    if (params.normalize_z_slices) {
        log.info "Normalizing Z-direction intensity drift (sigma=${params.znorm_smooth_sigma} slices)"
        znorm_input = stack_output
            .combine(stack_metadata)
            .map { zarr, zip, png, annotated, name, n, ids_str -> tuple(zarr, name, n, ids_str) }
        normalize_z_intensity(znorm_input)
        final_stack_output = normalize_z_intensity.out
    } else {
        final_stack_output = stack_output
    }

    // Stage 9: Report Generation (optional)
    if (params.generate_report) {
        generate_report(final_stack_output, subject_name)
    }

    // Stage 10: Atlas Registration (optional)
    if (params.align_to_ras_enabled) {
        log.info "Registering to Allen Mouse Brain Atlas (RAS alignment)"
        align_to_ras(final_stack_output, subject_name)
    }

    // Stage 11: Diagnostic Analyses (optional; enable via diagnostic_mode or individual flags)
    def runRotationAnalysis = params.diagnostic_mode || params.analyze_rotation_drift
    def runMotorOnlyStitch = params.diagnostic_mode || params.motor_only_stitch
    def runMotorOnlyStack = params.diagnostic_mode || params.motor_only_stack
    def runAcquisitionRotation = params.diagnostic_mode || params.analyze_acquisition_rotation

    if (params.diagnostic_mode) {
        log.info "DIAGNOSTIC MODE enabled:"
        log.info "  - Acquisition rotation analysis"
        log.info "  - Registration rotation drift"
        log.info "  - Motor-only stitching (per-slice)"
        log.info "  - Motor-only stacking (3D volume)"
    }

    if (runAcquisitionRotation) {
        analyze_acquisition_rotation(shifts_xy, register_pairwise.out.collect())
    }

    if (runRotationAnalysis) {
        analyze_rotation_drift(register_pairwise.out.collect())
    }

    if (runMotorOnlyStitch) {
        stitch_motor_only(illum_fixed)
    }

    if (runMotorOnlyStack) {
        motor_only_stack_input = normalize.out.normalized
            .map { slice_id, file -> file }
            .collect()
        stack_motor_only(motor_only_stack_input, shifts_xy)
    }

    // Compare motor-only vs refined stitching
    def runStitchingComparison = params.compare_stitching || params.diagnostic_mode
    if (runStitchingComparison) {
        log.info "Running stitching comparison (motor-only vs refined)..."

        stitch_refined(illum_fixed)

        if (!runMotorOnlyStitch) {
            stitch_motor_only(illum_fixed)
        }

        motor_stitch_with_id = stitch_motor_only.out.map { toSliceTuple(it) }
        refined_stitch_with_id = stitch_refined.out[0].map { toSliceTuple(it) }

        comparison_input = motor_stitch_with_id
            .combine(refined_stitch_with_id, by: 0)

        compare_stitching(comparison_input)
    }
}
