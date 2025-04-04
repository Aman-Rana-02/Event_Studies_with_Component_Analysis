import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
# Surpress pandas performance warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas')


def construct_event_windows(event_df, return_df, start_window=-45, return_window=45, min_periods=8):
    """
    Constructs event study windows around each event date.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event dates. Must have a 'Date' column.
    return_df : pandas.DataFrame
        DataFrame containing returns data, with dates as index and a 'Log Return' column.
    start_window : int, default=-45
        Starting point of the event window (negative days before event).
    return_window : int, default=45
        Ending point of the event window (positive days after event).
    min_periods : int, default=8
        Minimum number of periods required to calculate the mean for demeaning.
        If this is None, no demeaning is applied.
        I include this argument because it is useful to removing innovation in cumulative market returns.
        
    Returns
    -------
    featured_events : pandas.DataFrame
        DataFrame with events and their corresponding return windows.
    return_cols : list
        List of column names containing return data.
    """
    # Create shifted return columns for each day in the window
    return_df_copy = return_df.copy()
    for i in range(start_window, return_window + 1):
        return_df_copy[f'Log Return t={i}'] = return_df_copy['Log Return'].shift(-i)
    return_df_copy = return_df_copy.dropna()
    
    # Calculate cumulative returns
    for t in range(start_window, return_window + 1):
        return_df_copy[f'Cumulative Log Return {t}'] = return_df_copy.loc[:, f'Log Return t={start_window}':f'Log Return t={t}'].sum(axis=1)
    
    # Clean up intermediate columns
    for i in range(start_window, return_window + 1):
        return_df_copy = return_df_copy.drop(columns=[f'Log Return t={i}'])
    
    # Merge events with returns
    featured_events = pd.merge_asof(event_df, return_df_copy, on='Date', direction='backward', 
                                    suffixes=('', '_y'), allow_exact_matches=False)
    
    # Demean cumulative returns to focus on abnormal returns
    return_cols = [f'Cumulative Log Return {t}' for t in range(start_window, return_window + 1)]
    if min_periods:
        for col in return_cols:
            featured_events[col] = (featured_events[col] - featured_events[col].expanding(min_periods=min_periods).mean())
    
    # Clean up column names for easier access
    featured_events = featured_events.dropna()
    featured_events = featured_events.rename(columns={col: col.replace('Cumulative Log Return ', '') 
                                                     for col in return_cols})
    return_cols = [col.replace('Cumulative Log Return ', '') for col in return_cols]
    
    return featured_events, return_cols


def add_pcs(event_df, return_cols, n_pcs=3):
    """
    Performs Principal Component Analysis on event window returns and adds PC loadings to the DataFrame.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event data with return windows.
    return_cols : list
        List of column names containing return data.
    n_pcs : int, default=3
        Number of principal components to extract.
        
    Returns
    -------
    event_df : pandas.DataFrame
        DataFrame with PC loadings added.
    pc_results : dict
        Dictionary containing PCA results (components, explained variance, etc.)
    """
    # Set up the PCA pipeline
    scaler = StandardScaler()
    pca = PCA(n_components=n_pcs)
    pipe = Pipeline([('scaler', scaler), ('pca', pca)])
    
    # Fit and transform
    pipe.fit(event_df[return_cols])
    pc_loadings = pipe.transform(event_df[return_cols])
    
    # Add PC loadings to the DataFrame
    event_df_copy = event_df.copy()
    for i in range(n_pcs):
        event_df_copy[f'PC{i+1}'] = pc_loadings[:, i]
    
    # Store results in a dictionary
    pc_results = {
        'components': pipe.named_steps['pca'].components_,
        'explained_variance_ratio': pipe.named_steps['pca'].explained_variance_ratio_,
        'singular_values': pipe.named_steps['pca'].singular_values_,
        'pipeline': pipe
    }
    
    return event_df_copy, pc_results


def add_ics(event_df, return_cols, n_ics=3):
    """
    Performs Independent Component Analysis on event window returns and adds IC loadings to the DataFrame.
    
    Parameters
    ----------
    event_df : pandas.DataFrame
        DataFrame containing event data with return windows.
    return_cols : list
        List of column names containing return data.
    n_ics : int, default=3
        Number of independent components to extract.
        
    Returns
    -------
    event_df : pandas.DataFrame
        DataFrame with IC loadings added.
    ic_results : dict
        Dictionary containing ICA results (components, etc.)
    """
    # Set up the ICA pipeline
    scaler = StandardScaler()
    ica = FastICA(n_components=n_ics, random_state=42)
    pipe = make_pipeline(scaler, ica)
    
    # Fit and transform
    pipe.fit(event_df[return_cols])
    ic_loadings = pipe.transform(event_df[return_cols])
    
    # Add IC loadings to the DataFrame
    event_df_copy = event_df.copy()
    for i in range(n_ics):
        event_df_copy[f'IC{i+1}'] = ic_loadings[:, i]
    
    # Store results in a dictionary
    ic_results = {
        'components': pipe.named_steps['fastica'].components_,
        'mixing': pipe.named_steps['fastica'].mixing_,
        'pipeline': pipe
    }
    
    return event_df_copy, ic_results