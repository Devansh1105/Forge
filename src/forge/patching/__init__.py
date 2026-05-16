"""Model patching entrypoints."""


def patch(*args, **kwargs):
    """Patch a model with Forge kernels.

    The concrete patching API will be designed during the repo v1 phase.
    """

    raise NotImplementedError("Forge patching API is not designed yet.")


def unpatch(*args, **kwargs):
    """Undo Forge model patching.

    The concrete patching API will be designed during the repo v1 phase.
    """

    raise NotImplementedError("Forge patching API is not designed yet.")
