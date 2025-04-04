import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_ca_loadings(return_cols, ca_results, title='Component Loadings', xlabel='Days from Event', 
                    y_label='Component Loadings', figsize=(10, 5), legend=None, save_path=None):
    """
    Plots component loadings from PCA or ICA.
    
    Parameters
    ----------
    return_cols : list
        List of column names containing return data.
    ca_results : dict
        Dictionary containing component analysis results.
    title : str, default='Component Loadings'
        Plot title.
    xlabel : str, default='Days from Event'
        X-axis label.
    y_label : str, default='Component Loadings'
        Y-axis label.
    figsize : tuple, default=(10, 5)
        Figure size.
    legend : list, default=None
        Legend labels.
    save_path : str, default=None
        Path to save the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    n_days = len(ca_results['components'][0])
    x_values = np.arange(n_days) - len(return_cols) // 2  # Center around event date
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, ca_results['components'].T)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvline(0, color='black', lw=0.5, ls='--')
    
    if legend:
        ax.legend(legend)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_conditional_cumulative_returns(event_df, return_cols, ca_col, n_bins=3, 
                                       title=None, xlabel='Days from Event', 
                                       ylabel='Cumulative Return', figsize=(10, 5),
                                       save_path=None):
    """
    Plots cumulative returns grouped by component loading values.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event data with component loadings.
    return_cols : list
        List of column names containing return data.
    ca_col : str
        Column name for the component loading to group by.
    n_bins : int, default=3
        Number of bins to divide the loadings into.
    title : str, default=None
        Plot title. If None, a default title is generated.
    xlabel : str, default='Days from Event'
        X-axis label.
    ylabel : str, default='Cumulative Return'
        Y-axis label.
    figsize : tuple, default=(10, 5)
        Figure size.
    save_path : str, default=None
        Path to save the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    if not title:
        title = f'Cumulative Returns by {ca_col}'
    
    # Create bins for the component values
    event_df['CA Bin'] = pd.qcut(event_df[ca_col], n_bins, labels=False)
    
    # Calculate mean returns for each bin
    mean_returns = event_df.groupby('CA Bin')[return_cols].mean()
    
    # Subtract returns at t=0 to align at the event date
    t0_col = return_cols[len(return_cols) // 2]  # Middle column = event date
    mean_returns = mean_returns.subtract(mean_returns[t0_col], axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x_values = np.arange(len(return_cols)) - len(return_cols) // 2  # Center around event date
    
    for i in range(n_bins):
        label = ['Low', 'Medium', 'High'][i] if n_bins == 3 else f'Bin {i}'
        ax.plot(x_values, mean_returns.iloc[i], label=label)
    
    ax.set_title(title)
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_time_series(event_df, date_col, ts_cols, title='Time Series', 
                    xlabel='Date', ylabel='Value', figsize=(10, 5),
                    rolling_window=None, abs=False, save_path=None):
    """
    Plots time series of component loadings.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event data with component loadings.
    date_col : str
        Column name for dates.
    ts_cols : list
        List of column names to plot.
    title : str, default='Time Series'
        Plot title.
    xlabel : str, default='Date'
        X-axis label.
    ylabel : str, default='Value'
        Y-axis label.
    figsize : tuple, default=(10, 5)
        Figure size.
    rolling_window : int, default=None
        Window size for rolling average. If None, no rolling average is applied.
    abs : bool, default=False
        Whether to plot absolute values.
    save_path : str, default=None
        Path to save the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in ts_cols:
        data = event_df[col]
        if abs:
            data = data.abs()
        if rolling_window:
            data = data.rolling(window=rolling_window).mean()
        ax.plot(event_df[date_col], data, label=col)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_conditional_means(event_df, return_cols, group_col, y_cols, 
                          title='Conditional Means', xlabel='Group', 
                          ylabel='Mean Value', figsize=(10, 5), save_path=None):
    """
    Creates boxplots showing the distribution of component loadings by group.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event data with component loadings.
    return_cols : list
        List of column names containing return data.
    group_col : str
        Column name to group by.
    y_cols : list
        List of column names to plot.
    title : str, default='Conditional Means'
        Plot title.
    xlabel : str, default='Group'
        X-axis label.
    ylabel : str, default='Mean Value'
        Y-axis label.
    figsize : tuple, default=(10, 5)
        Figure size.
    save_path : str, default=None
        Path to save the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reshape data for boxplot
    plot_data = []
    labels = []
    
    for col in y_cols:
        for group in sorted(event_df[group_col].unique()):
            data = event_df[event_df[group_col] == group][col]
            plot_data.append(data)
            labels.append(f"{col}_{group}")
    
    ax.boxplot(plot_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig