class ProfileError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class TransformContractError(Exception):
    """
    A user-defined transform or per-voter function broke the contract of ``perturb()``.
    """

    def __init__(self, message: str):
        super().__init__(message)
