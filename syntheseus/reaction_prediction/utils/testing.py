def are_single_step_models_installed():
    """Check whether the single-step models can be imported successfully."""
    try:
        # Try to import the single-step model repositories to check if they are installed. It could
        # be the case that these are installed but their dependencies are not, in which case trying
        # to *use* the models would fail; nevertheless, the below is good enough for our usecases.

        import chemformer  # noqa: F401
        import graph2edits  # noqa: F401
        import local_retro  # noqa: F401
        import megan  # noqa: F401
        import mhnreact  # noqa: F401
        import root_aligned  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False
