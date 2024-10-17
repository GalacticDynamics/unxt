"""Copyright (c) 2023 unxt maintainers. All rights reserved."""

import os
from typing import Final

RUNTIME_TYPECHECKER: Final[str | None] = (
    v
    if (v := os.environ.get("UNXT_ENABLE_RUNTIME_TYPECHECKING", None)) != "None"
    else None
)
"""Runtime type checking variable "UNXT_ENABLE_RUNTIME_TYPECHECKING".

Set to "None" to disable runtime typechecking (default). Set to
"beartype.beartype" to enable runtime typechecking.

See https://docs.kidger.site/jaxtyping/api/runtime-type-checking for more
information on options.

"""
