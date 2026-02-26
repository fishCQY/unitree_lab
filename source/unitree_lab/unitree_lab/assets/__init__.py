# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

ISAAC_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

from .robots import *  # noqa: F401, F403
