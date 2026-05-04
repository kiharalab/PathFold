def test_package_imports():
    import pathfold
    from pathfold.diffusion import PathFoldFrameDiffusion
    from pathfold.inference import run_inference

    assert pathfold is not None
    assert PathFoldFrameDiffusion is not None
    assert run_inference is not None
