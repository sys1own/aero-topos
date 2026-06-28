# -*- coding: utf-8 -*-
"""Enable ``python -m blueprint_lang``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
