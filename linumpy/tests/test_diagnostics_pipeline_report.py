"""Direct-import tests for ``linumpy.diagnostics.pipeline_report``.

Exercises the extracted diagnostics library directly (no ``importlib`` script
loading), covering the same pure-helper behavior locked by
``scripts/tests/analysis/test_generate_pipeline_report.py`` plus additional
coverage for aggregation/formatting/discovery helpers (Plan 07.1-08).
"""

import json

import pytest

from linumpy.diagnostics.pipeline_report import (
    _render_grouped_issues_html,
    _render_interpolation_section_html,
    _render_interpolation_section_text,
    collect_issues,
    compute_cross_slice_trends,
    compute_overall_status,
    discover_diagnostic_data,
    discover_images,
    discover_interpolation_data,
    discover_slice_config_summary,
    extract_slice_id,
    format_value,
    generate_html_report,
    generate_sparkline_svg,
    generate_text_report,
    generate_trend_line_svg,
    generate_zip_bundle,
    get_status_color,
    get_status_emoji,
    get_step_status,
    group_issues,
    image_to_data_uri,
    parse_issue,
    render_image_gallery_html,
    separate_metrics_by_type,
    slug,
    sort_steps,
)

# ---------------------------------------------------------------------------
# Status / formatting helpers
# ---------------------------------------------------------------------------


def test_get_status_color_known_statuses():
    assert get_status_color("ok") == "#28a745"
    assert get_status_color("error") == "#dc3545"
    assert get_status_color("warning") == "#ffc107"
    assert get_status_color("info") == "#17a2b8"
    assert get_status_color("unknown") == "#6c757d"


def test_get_status_color_falls_back_to_unknown():
    assert get_status_color("nonsense") == "#6c757d"


def test_get_status_emoji_known_statuses():
    assert get_status_emoji("ok") == "✓"
    assert get_status_emoji("error") == "✗"
    assert get_status_emoji("warning") == "⚠"
    assert get_status_emoji("info") == "ℹ"
    assert get_status_emoji("unknown") == "?"


def test_get_status_emoji_falls_back_to_question():
    assert get_status_emoji("nonsense") == "?"


def test_format_value_scientific_for_small():
    assert "e" in format_value(0.00001)


def test_format_value_scientific_for_large():
    assert "e" in format_value(100000.0)


def test_format_value_fixed_for_normal_float():
    assert format_value(3.5) == "3.5000"


def test_format_value_long_list_summarised():
    out = format_value(list(range(10)))  # ty: ignore[invalid-argument-type]
    assert out == "[10 items]"


def test_format_value_short_list_preserved():
    out = format_value([1, 2, 3])  # ty: ignore[invalid-argument-type]
    assert out == "[1, 2, 3]"


def test_format_value_non_numeric_passthrough():
    assert format_value("hello") == "hello"  # ty: ignore[invalid-argument-type]


def test_format_value_int_passthrough():
    assert format_value(42) == "42"


def test_slug_normalizes():
    assert slug("Slice Quality Assessment!") == "slice-quality-assessment"


def test_slug_strips_leading_trailing_dashes():
    assert slug("---Foo---") == "foo"


def test_extract_slice_id_from_path():
    assert extract_slice_id("/data/slice_z03/tile.json") == "slice_z03"


def test_extract_slice_id_z_prefix():
    assert extract_slice_id("/data/z12/tile.json") == "z12"


def test_extract_slice_id_falls_back_to_stem():
    assert extract_slice_id("/data/no_match_here.json") == "no_match_here"


def test_sort_steps_orders_known_steps_first():
    aggregated = {"stack_slices": [1], "slice_quality_assessment": [1], "unknown_step": [1]}
    ordered = list(sort_steps(aggregated).keys())
    assert ordered == ["slice_quality_assessment", "stack_slices", "unknown_step"]


def test_sort_steps_unknown_sorted_alphabetically_after_known():
    aggregated = {"zzz_unknown": [1], "aaa_unknown": [1], "stack_slices": [1]}
    ordered = list(sort_steps(aggregated).keys())
    # known step first, then unknowns alphabetically
    assert ordered == ["stack_slices", "aaa_unknown", "zzz_unknown"]


# ---------------------------------------------------------------------------
# Issue parsing / grouping
# ---------------------------------------------------------------------------


def test_parse_issue_extracts_value_and_threshold():
    parsed = parse_issue("slice_z00: ssim: 0.5 < 0.8 (warning)")
    assert parsed["value"] == 0.5
    assert parsed["threshold"] == 0.8
    assert parsed["op"] == "<"
    assert parsed["source"] == "slice_z00"
    assert parsed["metric"] == "ssim"


def test_parse_issue_greater_equal_op():
    parsed = parse_issue("slice_z00: drift: 1.5 >= 1.0 (error)")
    assert parsed["value"] == 1.5
    assert parsed["op"] == ">="
    assert parsed["threshold"] == 1.0


def test_parse_issue_no_match_returns_none_values():
    parsed = parse_issue("slice_z00: metric_name: freeform text without numbers")
    assert parsed["value"] is None
    assert parsed["threshold"] is None
    assert parsed["source"] == "slice_z00"
    assert parsed["metric"] == "metric_name"


def test_parse_issue_too_few_parts():
    parsed = parse_issue("only_one_part")
    assert parsed["source"] == "only_one_part"
    assert parsed["metric"] == ""


def test_parse_issue_empty_string():
    parsed = parse_issue("")
    assert parsed["source"] == ""


def test_group_issues_groups_by_metric():
    issues = [
        "slice_z00: ssim: 0.5 < 0.8 (warning)",
        "slice_z01: ssim: 0.6 < 0.8 (warning)",
    ]
    grouped = group_issues(issues)
    assert len(grouped) == 1
    assert grouped[0]["count"] == 2
    assert grouped[0]["metric"] == "ssim"
    assert len(grouped[0]["values"]) == 2


def test_group_issues_multiple_metrics():
    issues = [
        "slice_z00: ssim: 0.5 < 0.8 (warning)",
        "slice_z00: drift: 1.5 >= 1.0 (error)",
    ]
    grouped = group_issues(issues)
    metrics = {g["metric"] for g in grouped}
    assert metrics == {"ssim", "drift"}


def test_group_issues_unparseable_goes_to_other():
    grouped = group_issues(["freeform text no numbers"])
    assert len(grouped) == 1
    assert grouped[0]["metric"] == ""


def test_group_issues_empty_list():
    assert group_issues([]) == []


# ---------------------------------------------------------------------------
# Metric separation
# ---------------------------------------------------------------------------


def test_separate_metrics_by_type():
    metrics_list = [
        {
            "metrics": {
                "ssim": {"value": 0.9, "status": "ok", "unit": ""},
                "resolution": {"value": 3.5, "status": "info", "unit": "um"},
            }
        }
    ]
    quality, info = separate_metrics_by_type(metrics_list)
    assert "ssim" in quality
    assert info["resolution"]["is_constant"] is True


def test_separate_metrics_by_type_non_constant_info():
    metrics_list = [
        {"metrics": {"resolution": {"value": 3.5, "status": "info", "unit": "um"}}},
        {"metrics": {"resolution": {"value": 4.5, "status": "info", "unit": "um"}}},
    ]
    _, info = separate_metrics_by_type(metrics_list)
    assert info["resolution"]["is_constant"] is False


def test_separate_metrics_by_type_non_dict_data_skipped():
    metrics_list = [{"metrics": {"bad": "not a dict"}}]
    quality, info = separate_metrics_by_type(metrics_list)
    assert quality == {}
    assert info == {}


def test_separate_metrics_by_type_string_values_constant():
    metrics_list = [
        {"metrics": {"name": {"value": "abc", "status": "info", "unit": ""}}},
        {"metrics": {"name": {"value": "abc", "status": "info", "unit": ""}}},
    ]
    _, info = separate_metrics_by_type(metrics_list)
    assert info["name"]["is_constant"] is True


def test_separate_metrics_by_type_empty():
    quality, info = separate_metrics_by_type([])
    assert quality == {}
    assert info == {}


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------


def test_generate_sparkline_svg_returns_svg_for_numeric():
    svg = generate_sparkline_svg([1.0, 2.0, 3.0, 4.0])
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "<rect" in svg


def test_generate_sparkline_svg_empty_for_single_value():
    assert generate_sparkline_svg([1.0]) == ""


def test_generate_sparkline_svg_empty_for_no_numeric():
    assert generate_sparkline_svg(["a", "b"]) == ""


def test_generate_sparkline_svg_with_statuses():
    svg = generate_sparkline_svg([1.0, 2.0, 3.0], statuses=["ok", "warning", "error"])
    assert "#28a745" in svg  # ok color
    assert "#dc3545" in svg  # error color


def test_generate_trend_line_svg_returns_svg():
    svg = generate_trend_line_svg([1.0, 2.0, 3.0, 4.0, 5.0])
    assert svg.startswith("<svg")
    assert "<polyline" in svg


def test_generate_trend_line_svg_with_trend_line():
    svg = generate_trend_line_svg([1.0, 2.0, 3.0, 4.0, 5.0], show_trend=True)
    assert "<line" in svg  # trend line present


def test_generate_trend_line_svg_no_trend_when_disabled():
    svg = generate_trend_line_svg([1.0, 2.0, 3.0, 4.0, 5.0], show_trend=False)
    assert "<line" not in svg


def test_generate_trend_line_svg_empty_for_single_value():
    assert generate_trend_line_svg([1.0]) == ""


# ---------------------------------------------------------------------------
# Cross-slice trends
# ---------------------------------------------------------------------------


def test_compute_cross_slice_trends_pairwise_registration():
    aggregated = {
        "pairwise_registration": [
            {"source_file": "a", "metrics": {"translation_x": {"value": 1.0}, "translation_y": {"value": 0.5}}},
            {"source_file": "b", "metrics": {"translation_x": {"value": 2.0}, "translation_y": {"value": 1.0}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert "registration_drift" in trends
    series_names = {s["name"] for s in trends["registration_drift"]["series"]}
    assert "Cumulative tx (px)" in series_names
    assert "Cumulative ty (px)" in series_names


def test_compute_cross_slice_trends_pairwise_with_rotation():
    aggregated = {
        "pairwise_registration": [
            {"source_file": "a", "metrics": {"rotation": {"value": 0.1}}},
            {"source_file": "b", "metrics": {"rotation": {"value": 0.2}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    series_names = {s["name"] for s in trends["registration_drift"]["series"]}
    assert "Cumulative rotation (deg)" in series_names


def test_compute_cross_slice_trends_xy_transform():
    aggregated = {
        "xy_transform_estimation": [
            {"source_file": "a", "metrics": {"transform_00": {"value": 1.0}, "transform_11": {"value": 1.0}}},
            {"source_file": "b", "metrics": {"transform_00": {"value": 1.1}, "transform_11": {"value": 1.1}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert "xy_transform" in trends


def test_compute_cross_slice_trends_crop_interface():
    aggregated = {
        "crop_interface": [
            {"source_file": "a", "metrics": {"detected_interface_depth_um": {"value": 100.0}}},
            {"source_file": "b", "metrics": {"detected_interface_depth_um": {"value": 110.0}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert "interface_depth" in trends


def test_compute_cross_slice_trends_normalize_intensities():
    aggregated = {
        "normalize_intensities": [
            {"source_file": "a", "metrics": {"mean_background": {"value": 50.0}}},
            {"source_file": "b", "metrics": {"mean_background": {"value": 55.0}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert "background_drift" in trends


def test_compute_cross_slice_trends_empty():
    assert compute_cross_slice_trends({}) == {}


def test_compute_cross_slice_trends_step_with_no_numeric_values():
    aggregated = {
        "pairwise_registration": [
            {"source_file": "a", "metrics": {"translation_x": {"value": "n/a"}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    assert trends == {}


def test_compute_cross_slice_trends_sorts_by_source_file():
    aggregated = {
        "pairwise_registration": [
            {"source_file": "z01", "metrics": {"translation_x": {"value": 10.0}}},
            {"source_file": "z00", "metrics": {"translation_x": {"value": 1.0}}},
        ]
    }
    trends = compute_cross_slice_trends(aggregated)
    tx_series = next(s for s in trends["registration_drift"]["series"] if "tx" in s["name"])
    # sorted by source file: z00 first → value 1.0, then z01 → 10.0; cumsum → [1.0, 11.0]
    assert list(tx_series["values"]) == [1.0, 11.0]


# ---------------------------------------------------------------------------
# Overall status / step status / issue collection
# ---------------------------------------------------------------------------


def test_compute_overall_status_counts():
    aggregated = {
        "step_a": [{"overall_status": "ok"}, {"overall_status": "warning"}],
        "step_b": [{"overall_status": "error"}],
    }
    all_statuses, errors, warnings, ok = compute_overall_status(aggregated)
    assert errors == 1
    assert warnings == 1
    assert ok == 1
    assert all_statuses == ["ok", "warning", "error"]


def test_compute_overall_status_empty():
    all_statuses, errors, warnings, ok = compute_overall_status({})
    assert all_statuses == []
    assert errors == 0
    assert warnings == 0
    assert ok == 0


def test_compute_overall_status_unknown_default():
    aggregated = {"step_a": [{}]}  # no overall_status key
    all_statuses, errors, warnings, ok = compute_overall_status(aggregated)
    assert all_statuses == ["unknown"]
    assert errors == 0
    assert warnings == 0
    assert ok == 0


def test_get_step_status_prioritizes_error():
    assert get_step_status([{"overall_status": "ok"}, {"overall_status": "error"}]) == "error"


def test_get_step_status_warning_when_no_error():
    assert get_step_status([{"overall_status": "ok"}, {"overall_status": "warning"}]) == "warning"


def test_get_step_status_ok_when_all_ok():
    assert get_step_status([{"overall_status": "ok"}, {"overall_status": "ok"}]) == "ok"


def test_get_step_status_unknown_default():
    assert get_step_status([{}]) == "ok"  # no error/warning → falls through to ok


def test_collect_issues_prefixes_source():
    metrics_list = [{"source_file": "/x/slice_z00.json", "warnings": ["low ssim"], "errors": []}]
    warnings, errors = collect_issues(metrics_list)
    assert warnings == ["slice_z00: low ssim"]
    assert errors == []


def test_collect_issues_collects_errors():
    metrics_list = [{"source_file": "slice_z01.json", "warnings": [], "errors": ["bad drift"]}]
    warnings, errors = collect_issues(metrics_list)
    assert warnings == []
    assert errors == ["slice_z01: bad drift"]


def test_collect_issues_empty():
    warnings, errors = collect_issues([])
    assert warnings == []
    assert errors == []


def test_collect_issues_missing_source_uses_unknown():
    warnings, _errors = collect_issues([{"warnings": ["w"], "errors": []}])
    assert warnings == ["unknown: w"]


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def test_render_grouped_issues_html_single_issue():
    grouped = [
        {"metric": "ssim", "count": 1, "values": [0.5], "threshold": 0.8, "op": "<", "details": ["slice_z00: ssim: 0.5 < 0.8"]}
    ]
    html = _render_grouped_issues_html(grouped, "error", "Errors")
    assert "Errors" in html
    assert "slice_z00: ssim: 0.5 < 0.8" in html
    assert "<strong>1</strong>" not in html  # count badge is plain


def test_render_grouped_issues_html_multiple_issues():
    grouped = [
        {
            "metric": "ssim",
            "count": 3,
            "values": [0.5, 0.6, 0.7],
            "threshold": 0.8,
            "op": "<",
            "details": ["slice_z00: ssim: 0.5 < 0.8", "slice_z01: ssim: 0.6 < 0.8", "slice_z02: ssim: 0.7 < 0.8"],
        }
    ]
    html = _render_grouped_issues_html(grouped, "warning", "Warnings")
    assert "ssim" in html
    assert "3 slices affected" in html
    assert "threshold: 0.8" in html


def test_render_grouped_issues_html_no_threshold():
    grouped = [{"metric": "", "count": 2, "values": [], "threshold": None, "op": ">", "details": ["d1", "d2"]}]
    html = _render_grouped_issues_html(grouped, "error", "Errors")
    assert "2 occurrences" in html


def test_render_interpolation_section_html_smoke():
    interpolation = {
        "summary": {
            "count": 2,
            "n_failed": 1,
            "n_succeeded": 1,
            "method_counts": {"zmorph": 2},
            "method_used_counts": {"zmorph": 1},
            "fallback_counts": {"low_ncc": 1},
            "pre_reg_ncc_mean": 0.5,
            "post_reg_ncc_mean": 0.8,
            "ncc_improvement_mean": 0.3,
        },
        "rows": [
            {
                "slice_id": "z00",
                "method_used": "zmorph",
                "interpolation_failed": False,
                "pre_reg_ncc": 0.5,
                "post_reg_ncc": 0.8,
            },
            {
                "slice_id": "z01",
                "method_used": "",
                "interpolation_failed": True,
                "fallback_reason": "low_ncc",
                "pre_reg_ncc": 0.1,
                "post_reg_ncc": 0.2,
            },
        ],
        "images": [],
        "slice_config_final": None,
    }
    html = _render_interpolation_section_html(interpolation)
    assert "Slice Interpolation" in html
    assert "Gaps Detected" in html


def test_render_interpolation_section_html_all_failed_is_error():
    interpolation = {
        "summary": {"count": 2, "n_failed": 2, "n_succeeded": 0},
        "rows": [],
        "images": [],
    }
    html = _render_interpolation_section_html(interpolation)
    # error color used when all fail
    assert "#dc3545" in html


def test_render_interpolation_section_html_some_failed_is_warning():
    interpolation = {
        "summary": {"count": 2, "n_failed": 1, "n_succeeded": 1},
        "rows": [],
        "images": [],
    }
    html = _render_interpolation_section_html(interpolation)
    assert "#ffc107" in html  # warning color


def test_render_interpolation_section_text_smoke():
    interpolation = {
        "summary": {
            "count": 2,
            "n_failed": 1,
            "n_succeeded": 1,
            "method_used_counts": {"zmorph": 1},
            "fallback_counts": {"low_ncc": 1},
            "pre_reg_ncc_mean": 0.5,
            "post_reg_ncc_mean": 0.8,
            "ncc_improvement_mean": 0.3,
        },
        "rows": [
            {
                "slice_id": "z00",
                "method_used": "zmorph",
                "interpolation_failed": False,
                "pre_reg_ncc": 0.5,
                "post_reg_ncc": 0.8,
            },
        ],
    }
    text = _render_interpolation_section_text(interpolation)
    assert "SLICE INTERPOLATION" in text
    assert "Gaps detected" in text
    assert "zmorph: 1" in text
    assert "low_ncc: 1" in text


def test_render_interpolation_section_text_empty():
    text = _render_interpolation_section_text({})
    assert "SLICE INTERPOLATION" in text
    assert "Gaps detected          : 0" in text


def test_render_interpolation_section_text_many_rows_truncated():
    rows = [{"slice_id": f"z{i:02d}", "method_used": "zmorph", "interpolation_failed": False} for i in range(60)]
    text = _render_interpolation_section_text({"summary": {"count": 60}, "rows": rows})
    assert "more row(s) not shown" in text


# ---------------------------------------------------------------------------
# Discovery functions (use tmp_path for filesystem fixtures)
# ---------------------------------------------------------------------------


def test_discover_interpolation_data_none_when_dir_missing(tmp_path):
    assert discover_interpolation_data(tmp_path) is None


def test_discover_interpolation_data_none_when_no_diag_files(tmp_path):
    (tmp_path / "interpolate_missing_slice").mkdir()
    assert discover_interpolation_data(tmp_path) is None


def test_discover_interpolation_data_reads_json(tmp_path):
    interp_dir = tmp_path / "interpolate_missing_slice"
    interp_dir.mkdir()
    diag = {
        "slice_id": "z00",
        "method": "zmorph",
        "method_used": "zmorph",
        "interpolation_failed": False,
        "pre_reg_ncc": 0.5,
        "post_reg_ncc": 0.8,
        "ncc_improvement": 0.3,
    }
    (interp_dir / "slice_z00_interpolated_diagnostics.json").write_text(json.dumps(diag))
    result = discover_interpolation_data(tmp_path)
    assert result is not None
    assert result["summary"]["count"] == 1
    assert result["summary"]["n_succeeded"] == 1
    assert result["summary"]["n_failed"] == 0
    assert result["rows"][0]["slice_id"] == "z00"


def test_discover_interpolation_data_failed_slice(tmp_path):
    interp_dir = tmp_path / "interpolate_missing_slice"
    interp_dir.mkdir()
    diag = {
        "slice_id": "z01",
        "method": "zmorph",
        "interpolation_failed": True,
        "fallback_reason": "low_ncc",
    }
    (interp_dir / "slice_z01_interpolated_diagnostics.json").write_text(json.dumps(diag))
    result = discover_interpolation_data(tmp_path)
    assert result is not None
    assert result["summary"]["n_failed"] == 1
    assert result["summary"]["n_succeeded"] == 0
    assert result["summary"]["fallback_counts"] == {"low_ncc": 1}


def test_discover_interpolation_data_skips_invalid_json(tmp_path):
    interp_dir = tmp_path / "interpolate_missing_slice"
    interp_dir.mkdir()
    (interp_dir / "slice_z00_interpolated_diagnostics.json").write_text("not valid json{")
    assert discover_interpolation_data(tmp_path) is None


def test_discover_interpolation_data_preview_images(tmp_path):
    interp_dir = tmp_path / "interpolate_missing_slice"
    interp_dir.mkdir()
    (interp_dir / "slice_z00_interpolated_diagnostics.json").write_text(json.dumps({"slice_id": "z00"}))
    (interp_dir / "slice_z00_interpolated_preview.png").write_bytes(b"\x89PNG fake")
    result = discover_interpolation_data(tmp_path)
    assert result is not None
    assert len(result["images"]) == 1


def test_discover_diagnostic_data_empty_when_no_dir(tmp_path):
    assert discover_diagnostic_data(tmp_path) == {}


def test_discover_diagnostic_data_reads_known_subdir(tmp_path):
    diag_dir = tmp_path / "diagnostics" / "rotation_analysis"
    diag_dir.mkdir(parents=True)
    (diag_dir / "data.json").write_text(json.dumps({"rotation": 1.5}))
    result = discover_diagnostic_data(tmp_path)
    assert "rotation_analysis" in result
    assert result["rotation_analysis"]["label"] == "Rotation Drift Analysis"
    assert len(result["rotation_analysis"]["json_data"]) == 1


def test_discover_diagnostic_data_collects_pngs(tmp_path):
    diag_dir = tmp_path / "diagnostics" / "rotation_analysis"
    diag_dir.mkdir(parents=True)
    (diag_dir / "plot.png").write_bytes(b"\x89PNG fake")
    result = discover_diagnostic_data(tmp_path)
    assert len(result["rotation_analysis"]["images"]) == 1


def test_discover_diagnostic_data_skips_invalid_json(tmp_path):
    diag_dir = tmp_path / "diagnostics" / "rotation_analysis"
    diag_dir.mkdir(parents=True)
    (diag_dir / "bad.json").write_text("invalid{")
    result = discover_diagnostic_data(tmp_path)
    # subdir present but no valid json → not included (json_data empty, no images)
    assert "rotation_analysis" not in result


def test_discover_diagnostic_data_skips_unknown_subdir(tmp_path):
    diag_dir = tmp_path / "diagnostics" / "unknown_subdir"
    diag_dir.mkdir(parents=True)
    (diag_dir / "data.json").write_text(json.dumps({"x": 1}))
    result = discover_diagnostic_data(tmp_path)
    assert "unknown_subdir" not in result


def test_discover_slice_config_summary_none_when_missing(tmp_path):
    assert discover_slice_config_summary(tmp_path) is None


def test_discover_slice_config_summary_reads_top_final(tmp_path):
    csv = tmp_path / "slice_config_final.csv"
    csv.write_text(
        "slice_id,use,auto_excluded,interpolated,interpolation_failed,rehomed,rehoming_reliable,quality_score,exclude_reason\n"
        "z00,true,false,false,false,false,1,0.9,\n"
        "z01,false,true,false,false,true,0,0.3,low_quality\n"
    )
    result = discover_slice_config_summary(tmp_path)
    assert result is not None
    assert result["n_total"] == 2
    assert result["n_use_true"] == 1
    assert result["n_use_false"] == 1
    assert result["n_auto_excluded"] == 1
    assert result["n_rehomed"] == 1
    assert result["n_rehoming_unreliable"] == 1
    assert result["reasons"] == {"low_quality": 1}
    assert result["quality_score_mean"] == pytest.approx(0.6)


def test_discover_images_empty_when_no_dir(tmp_path):
    images = discover_images(tmp_path)
    assert images["overview"] == []
    assert images["stitch_preview"] == []
    assert images["common_space_preview"] == []


def test_discover_images_stitch_previews(tmp_path):
    stitch_dir = tmp_path / "previews" / "stitched_slices"
    stitch_dir.mkdir(parents=True)
    (stitch_dir / "slice_z00.png").write_bytes(b"\x89PNG")
    (stitch_dir / "slice_z01.png").write_bytes(b"\x89PNG")
    images = discover_images(tmp_path)
    assert len(images["stitch_preview"]) == 2


def test_discover_images_common_space_previews(tmp_path):
    cs_dir = tmp_path / "common_space_previews"
    cs_dir.mkdir(parents=True)
    (cs_dir / "cs_00.png").write_bytes(b"\x89PNG")
    images = discover_images(tmp_path)
    assert len(images["common_space_preview"]) == 1


def test_discover_images_overview_from_stack_dir(tmp_path):
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    (stack_dir / "overview.png").write_bytes(b"\x89PNG")
    images = discover_images(tmp_path)
    assert len(images["overview"]) == 1


def test_discover_images_overview_from_cli_arg(tmp_path):
    png = tmp_path / "cli_overview.png"
    png.write_bytes(b"\x89PNG")
    images = discover_images(tmp_path, overview_png=png)
    assert images["overview"] == [png]


def test_discover_images_fix_illum_previews(tmp_path):
    diag_dir = tmp_path / "fix_illumination_basic" / "diagnostics"
    diag_dir.mkdir(parents=True)
    (diag_dir / "basic_00.png").write_bytes(b"\x89PNG")
    images = discover_images(tmp_path)
    assert len(images["fix_illum_basic_preview"]) == 1


def test_discover_images_diag_subdir_category(tmp_path):
    diag_dir = tmp_path / "diagnostics" / "rotation_analysis"
    diag_dir.mkdir(parents=True)
    (diag_dir / "rot.png").write_bytes(b"\x89PNG")
    images = discover_images(tmp_path)
    assert "diag_rotation_analysis" in images
    assert len(images["diag_rotation_analysis"]) == 1


# ---------------------------------------------------------------------------
# Image encoding / gallery / zip
# ---------------------------------------------------------------------------


def test_image_to_data_uri_embeds_bytes(tmp_path):
    png = tmp_path / "test.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfakepngdata")
    uri = image_to_data_uri(png)
    assert uri.startswith("data:image/png;base64,")


def test_render_image_gallery_html_empty():
    assert render_image_gallery_html([]) == ""


def test_render_image_gallery_html_link_mode(tmp_path):
    png = tmp_path / "slice_z00.png"
    png.write_bytes(b"\x89PNG")
    html = render_image_gallery_html([png], mode="link", category="stitch_preview")
    assert "previews/stitch_preview/slice_z00.png" in html
    assert "slice_z00" in html


def test_generate_zip_bundle_creates_zip(tmp_path):
    import zipfile

    png = tmp_path / "img.png"
    png.write_bytes(b"\x89PNG")
    out = tmp_path / "report.zip"
    generate_zip_bundle("<html>report</html>", {"stitch_preview": [png]}, out)
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "index.html" in names
        assert "previews/stitch_preview/img.png" in names


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------


def test_generate_text_report_smoke():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "ok",
                "metrics": {"ssim": {"value": 0.9, "status": "ok", "unit": ""}},
                "warnings": [],
                "errors": [],
            }
        ]
    }
    report = generate_text_report(aggregated, title="Test Report")
    assert "Test Report" in report
    assert "SLICE QUALITY ASSESSMENT" in report
    assert "SUMMARY" in report


def test_generate_text_report_with_warnings_and_errors():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "error",
                "metrics": {"ssim": {"value": 0.5, "status": "error", "unit": ""}},
                "warnings": ["slice_z00: ssim: 0.5 < 0.8 (warning)"],
                "errors": ["slice_z00: drift: 1.5 >= 1.0 (error)"],
            }
        ]
    }
    report = generate_text_report(aggregated, title="Report")
    assert "ERRORS" in report
    assert "WARNINGS" in report


def test_generate_text_report_verbose():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "ok",
                "metrics": {"ssim": {"value": 0.9, "status": "ok", "unit": ""}},
                "warnings": [],
                "errors": [],
            }
        ]
    }
    report = generate_text_report(aggregated, title="Report", verbose=True)
    assert "Individual Results" in report
    assert "ssim" in report


def test_generate_text_report_with_slice_config_summary():
    aggregated = {
        "slice_quality_assessment": [
            {"source_file": "z00.json", "overall_status": "ok", "metrics": {}, "warnings": [], "errors": []}
        ]
    }
    sc = {
        "source": "/x/slice_config_final.csv",
        "n_total": 10,
        "n_use_true": 8,
        "n_use_false": 2,
        "n_auto_excluded": 1,
        "n_interpolated": 1,
        "n_interpolation_failed": 0,
        "n_rehomed": 0,
        "n_rehoming_unreliable": 0,
        "reasons": {"low_quality": 2},
        "quality_score_mean": 0.85,
        "quality_score_min": 0.3,
    }
    report = generate_text_report(aggregated, title="Report", slice_config_summary=sc)
    assert "SLICE CONFIGURATION" in report
    assert "low_quality=2" in report
    assert "mean=0.850" in report


def test_generate_text_report_with_interpolation():
    aggregated = {
        "slice_quality_assessment": [
            {"source_file": "z00.json", "overall_status": "ok", "metrics": {}, "warnings": [], "errors": []}
        ]
    }
    interp = {"summary": {"count": 1, "n_failed": 0, "n_succeeded": 1}, "rows": []}
    report = generate_text_report(aggregated, title="Report", interpolation=interp)
    assert "SLICE INTERPOLATION" in report


def test_generate_text_report_empty_aggregated():
    report = generate_text_report({}, title="Empty")
    assert "Empty" in report
    assert "End of Report" in report


def test_generate_html_report_smoke():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "ok",
                "metrics": {"ssim": {"value": 0.9, "status": "ok", "unit": ""}},
                "warnings": [],
                "errors": [],
            }
        ]
    }
    html = generate_html_report(aggregated, title="HTML Report")
    assert "<!DOCTYPE html>" in html
    assert "HTML Report" in html


def test_generate_html_report_with_errors():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "error",
                "metrics": {},
                "warnings": [],
                "errors": ["slice_z00: drift: 1.5 >= 1.0 (error)"],
            }
        ]
    }
    html = generate_html_report(aggregated, title="Report")
    assert "error" in html.lower()


def test_generate_html_report_with_warnings():
    aggregated = {
        "slice_quality_assessment": [
            {
                "source_file": "slice_z00.json",
                "overall_status": "warning",
                "metrics": {},
                "warnings": ["slice_z00: ssim: 0.5 < 0.8 (warning)"],
                "errors": [],
            }
        ]
    }
    html = generate_html_report(aggregated, title="Report")
    assert "warning" in html.lower()
