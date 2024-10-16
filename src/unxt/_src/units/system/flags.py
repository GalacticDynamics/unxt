class AbstractUnitSystemFlag:
    """Abstract class for unit system flags to provide dispatch control."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Never:
        msg = "unit system flag classes cannot be instantiated."
        raise ValueError(msg)


class StandardUnitSystemFlag(AbstractUnitSystemFlag):
    """Unit system flag to indicate a standard unit system with no additional args."""
