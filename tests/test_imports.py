def test_package_imports():
    import alphapathfold
    from alphapathfold.diffusion import AlphaPathFoldFrameDiffusion
    from alphapathfold.inference import run_inference

    assert alphapathfold is not None
    assert AlphaPathFoldFrameDiffusion is not None
    assert run_inference is not None
