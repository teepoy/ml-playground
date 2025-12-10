import alignment_methods


def test_public_api_is_exposed():
    assert callable(alignment_methods.ncc)
    assert callable(alignment_methods.phase_correlation)
    assert callable(alignment_methods.two_stage_correlation)


def test_correlation_methods_shim_points_to_split_modules():
    from alignment_methods import correlation_methods

    assert correlation_methods.ncc is alignment_methods.ncc
    assert correlation_methods.phase_correlation is alignment_methods.phase_correlation
    assert (
        correlation_methods.two_stage_correlation
        is alignment_methods.two_stage_correlation
    )
