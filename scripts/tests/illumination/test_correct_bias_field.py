def test_help(script_runner):
    ret = script_runner.run(["linum-correct-bias-field", "--help"])
    assert ret.success
    assert "--zero_mask_mode" in ret.stdout
    assert "--no-zero_outside_mask" in ret.stdout
    assert "mask_only" in ret.stdout


def test_mask_only_zeros_agarose_keeps_tissue(tmp_path, script_runner):
    """mask_only skips N4 but still drops the agarose halo."""
    import dask.array as da
    import numpy as np

    from linumpy.io.zarr import read_omezarr, save_omezarr

    rng = np.random.default_rng(0)
    vol = rng.random((8, 32, 32)).astype(np.float32) * 0.05  # agarose
    tissue = vol[:, 8:24, 8:24].copy()
    tissue += 0.7
    vol[:, 8:24, 8:24] = tissue
    inp = tmp_path / "in.ome.zarr"
    out = tmp_path / "out.ome.zarr"
    save_omezarr(
        da.from_array(vol),
        inp,
        voxel_size=(0.01, 0.01, 0.01),
        chunks=vol.shape,
        n_levels=0,
    )

    ret = script_runner.run(
        [
            "linum-correct-bias-field",
            str(inp),
            str(out),
            "--mode",
            "mask_only",
            "--zero_mask_mode",
            "silhouette",
            "--no-histogram_match",
            "--n_levels",
            "0",
            "--no_isotropic",
        ]
    )
    assert ret.success, ret.stderr
    result, _ = read_omezarr(out, level=0)
    arr = np.asarray(result[:], dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    # Halo gone; interior tissue kept.
    assert float(arr[:, 0, 0].max()) == 0.0
    assert float(arr[:, 16, 16].mean()) > 0.5
