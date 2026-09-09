def test_help(script_runner):
    ret = script_runner.run(["linum-correct-bias-field", "--help"])
    assert ret.success
    assert "--zero_mask_mode" in ret.stdout
    assert "--no-zero_outside_mask" in ret.stdout
