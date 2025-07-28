"""
Global plotting configuration for publication-quality PDF figures.
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def configure_plotting_for_pdf():
    """
    Configure matplotlib and seaborn for high-quality PDF output.
    This should be called at the beginning of any script that generates plots.
    """
    # Use non-interactive backend
    matplotlib.use('Agg')
    
    # Configure matplotlib for publication quality
    plt.rcParams.update({
        # Font settings
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'text.usetex': False,  # Set to True if LaTeX is available
        
        # Figure settings
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        
        # Axes settings
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.edgecolor': 'black',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        
        # Tick settings
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        
        # Line settings
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        
        # Error bar settings
        'errorbar.capsize': 3,
        
        # PDF-specific settings
        'pdf.fonttype': 42,  # Embed fonts as Type 42 (TrueType)
        'ps.fonttype': 42,   # For EPS compatibility
    })
    
    # Configure seaborn
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True
    })
    
    # Use a publication-friendly color palette
    sns.set_palette("husl", n_colors=8)
    
    print("Plotting configured for high-quality PDF output")


def get_figure_size(width_ratio: float = 1.0, height_ratio: float = 0.75) -> tuple:
    """
    Get figure size based on standard ratios.
    
    Args:
        width_ratio: Ratio of standard width (default 8 inches)
        height_ratio: Ratio relative to width
        
    Returns:
        Tuple of (width, height) in inches
    """
    base_width = 8.0
    width = base_width * width_ratio
    height = width * height_ratio
    return (width, height)


def get_subplot_layout(n_plots: int) -> tuple:
    """
    Get optimal subplot layout for n plots.
    
    Args:
        n_plots: Number of subplots
        
    Returns:
        Tuple of (rows, cols)
    """
    if n_plots <= 1:
        return (1, 1)
    elif n_plots <= 2:
        return (1, 2)
    elif n_plots <= 4:
        return (2, 2)
    elif n_plots <= 6:
        return (2, 3)
    elif n_plots <= 9:
        return (3, 3)
    else:
        # For more plots, use a more rectangular layout
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        return (rows, cols)


# Color schemes for different types of plots
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'correct': '#2ca02c',
    'incorrect': '#d62728',
    'uncertainty': '#9467bd',
    'ensemble': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

# Markers for different methods
MARKERS = {
    'ours': 'o',
    'baseline': 's',
    'comparison': '^',
    'theoretical': '--',
    'empirical': '-'
}

# Line styles
LINESTYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dashdot': '-.'
}
