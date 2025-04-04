# EventCA: Event Studies with Component Analysis

## Overview

**EventCA requires only a time series of returns and a list of event dates to perform sophisticated event studies.**

EventCA is a Python package that facilitates event studies using Principal Component Analysis (PCA) and Independent Component Analysis (ICA). The package provides tools to analyze market reactions around specific events, identify underlying patterns, and interpret results through powerful component analysis techniques.

Event studies are widely used in finance and economics to assess the impact of specific events on market prices. EventCA extends traditional event study methodology by incorporating component analysis to identify latent patterns in market reactions.

We include a casestudy around FOMC meetings which shows that the market typically prices in FOMC meetings, even though they are not observed. 


## Installation

```bash
# Clone the repository
git clone https://github.com/Aman-Rana-02/Event_Studies_w_CA.git
cd Event_Studies_w_CA

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Features

- **Event Window Construction**: Easily create event windows around specific dates
- **Component Analysis**: Apply PCA and ICA to identify patterns in market reactions
- **Visualization Tools**: Comprehensive plotting functions for component loadings, conditional returns, and time series analysis
- **Event Characteristic Analysis**: Examine how different event characteristics relate to market reactions
- **Extensibility**: All visualization functions return matplotlib figures that can be further customized

## Quick Start

```python
import pandas as pd
import numpy as np
from eventca.analysis import construct_event_windows, add_pcs, add_ics
from eventca.visualization import plot_ca_loadings, plot_conditional_cumulative_returns

# Load your data
returns_df = pd.read_csv('your_returns_data.csv')
returns_df['Date'] = pd.to_datetime(returns_df['Date'])
returns_df['Log Return'] = np.log(returns_df['Price']).diff()
returns_df.set_index('Date', inplace=True)

events_df = pd.read_csv('your_events_data.csv')
events_df['Date'] = pd.to_datetime(events_df['Date'])

# Perform analysis
featured_events, return_cols = construct_event_windows(
    event_df=events_df, 
    return_df=returns_df,
    start_window=-20,
    return_window=20
)

# Add PCA components
featured_events, pc_res = add_pcs(featured_events, return_cols, n_pcs=3)

# Visualize results
plot_ca_loadings(return_cols, pc_res, legend=['PC1', 'PC2', 'PC3'])
plot_conditional_cumulative_returns(featured_events, return_cols, 'PC1')
```

## Case Studies

The repository includes detailed case studies to demonstrate the package's capabilities:

1. **PCA vs. ICA**: Explanation and comparison of these component analysis techniques
2. **FOMC Meetings**: Analysis of S&P 500 market reactions to Federal Open Market Committee meetings
3. **Package Introduction**: Step-by-step walkthrough of the EventCA package capabilities

## Directory Structure

- `data/`: Raw datasets including S&P 500 returns and FOMC meeting dates
- `eventca/`: Source code for the package
  - `analysis.py`: Core analytical functions
  - `visualization.py`: Plotting and visualization tools
- `figs/`: Generated figures from the analysis
- `scripts/`: Jupyter notebooks demonstrating usage
  - `0.0.PCA_vs_ICA.ipynb`: Introduction to PCA vs ICA concepts
  - `1.0.PCA_vs_ICA_illustrative.ipynb`: Illustrative examples with simulated data
  - `2.0.FOMC_case_study.ipynb`: Detailed case study of FOMC meetings - WITHOUT EventCA
  - `3.0.introducing_eventca.ipynb`: Introduction to the EventCA package, much cleaner and more organized
- `references/`: Additional reference materials and example code

## Background

The EventCA package was developed to address the need for more sophisticated analysis of market reactions around significant events. Traditional event studies often focus on average cumulative abnormal returns, but this approach can miss important patterns in how markets react to different types of events.

By applying component analysis techniques, EventCA can identify:
- Different types of market reactions (e.g., anticipation, overreaction, reversal)
- How these reaction patterns have evolved over time
- Relationships between event characteristics and market response patterns

## Future Work

- Support for multiple return series analysis
- Integration with statistical testing frameworks
- Interactive visualizations
- Additional component analysis techniques

## Contact
