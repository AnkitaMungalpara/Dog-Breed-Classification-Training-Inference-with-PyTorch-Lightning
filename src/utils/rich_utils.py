from pathlib import Path
from typing import Sequence, Any, Dict

import rich
import rich.syntax
import rich.tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from rich.panel import Panel

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

console = Console()


def print_config_tree(
    config: Dict[str, Any],
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of config using Rich library and its tree structure."""

    style = "dim"
    tree = rich.tree
