#!/usr/bin/env python3
"""
TernaryLLM-20B Training Script
Main training entry point with comprehensive functionality for training ternary models.

Author: Zombie
Date: 2025-11-05
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import argparse
import logging
import traceback
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import multiprocessing as mp

# Third-party imports
try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.cuda.amp as amp
    from torch.profiler import profile, record_function, ProfilerActivity
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")

try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not available")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text