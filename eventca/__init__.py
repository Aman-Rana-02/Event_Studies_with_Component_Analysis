# Import key functions for easy access
from .analysis import construct_event_windows, add_pcs, add_ics
from .visualization import (
    plot_ca_loadings, 
    plot_conditional_cumulative_returns,
    plot_time_series,
    plot_conditional_means
)

# Version information
__version__ = '0.1.0'