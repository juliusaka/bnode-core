"""
Common plotting utilities shared across plotting modules.
"""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def save_figure(
    fig: plt.Figure,
    output_folder: Path,
    filename_base: str,
    file_type: Optional[str] = None,
    dpi: int = 300,
    bbox_inches: str = 'tight',
) -> None:
    """Save a figure with consistent settings.

    Args:
        fig: Matplotlib figure object to save
        output_folder: Path to save the figure to
        filename_base: Base filename without extension
        file_type: File type extension (e.g., 'png', 'pdf'). If None, defaults to 'png'.
        dpi: Resolution in dots per inch (applies to rasterized content and raster outputs)
        bbox_inches: Bounding box option passed to savefig
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Default and sanitize extension
    file_type = (file_type or 'png').lstrip('.')
    filename = f"{filename_base}.{file_type}"

    # Save with consistent options
    if file_type.lower() in ['pdf', 'svg', 'eps']:
        fig.savefig(output_folder / filename, dpi=dpi, bbox_inches=bbox_inches)
    else:
        fig.savefig(output_folder / filename, dpi=dpi, bbox_inches=bbox_inches)

    # Do not close here; caller may choose to close
