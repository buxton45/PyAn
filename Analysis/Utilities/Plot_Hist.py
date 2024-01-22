#!/usr/bin/env python

"""
Utilities specifically for plotting histograms
"""

#---------------------------------------------------------------------
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns, natsort_keygen
from copy import deepcopy
#---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import dates
import matplotlib.colors as mcolors
import matplotlib.cm as cm #e.g. for cmap=cm.jet
#---------------------------------------------------------------------
import Utilities
import Utilities_xml
import Utilities_df
import Utilities_dt
import Plot_General
#---------------------------------------------------------------------

#**************************************************
def adjust_patch_positions_and_widths(
    ax, 
    div_width_by, 
    position_idx, 
    i_patch_beg=None, 
    i_patch_end=None, 
    orient='v'
):
    r"""
    This is intended to shrink the width of histogram bins and shift them to allow multiple histograms
      to be plotted next to each other, instead of on top of each other.
      
    To be safe, this should probably be called before any legends or other objects are added to the axis
    
    NOTE: patches to be used are: [i_patch_beg, i_patch_end) (i.e., does not include i_patch_end)
          Also, if i_patch_beg is unset, i_patch_beg=0 will be used
                if i_patch_end is unset, i_patch_end=len(ax.patches) will be used
    
    Basically, the width of each bar will be reduced by a factor of div_width_by.
    After the division, the new bar can fit in any of div_width_by positions relative to the orignal bin edge.
    Which division to place the histogram is determined by position_idx
    
    Let's assume div_width_by = 3
    Assume that originally a single bar in the histogram occupies a width of 3 units
      i.e., bar_i = |_ _ _|
      After the division, the new bar_i can be position at:
        position_idx = 0: |_| _ _
        position_idx = 1: _ |_| _
        position_idx = 2: _ _ |_|
    
    """
    #-------------------------
    assert(orient=='v' or orient=='h')
    #-------------------------
    assert(position_idx>=0 and position_idx<div_width_by)
    if i_patch_beg is None:
        i_patch_beg = 0
    if i_patch_end is None:
        i_patch_end = len(ax.patches)
    #-------------------------
    # Note: the original x-position of the bar is defined as the left edge (top edge if horizontal orientation)
    patches = ax.patches[i_patch_beg:i_patch_end]
    for patch in patches:
        if orient=='v':
            new_width = patch.get_width()/div_width_by
            new_x = patch.get_x() + position_idx*new_width
            #-----
            patch.set_x(new_x)
            patch.set_width(new_width)
        else:
            new_height = patch.get_height()/div_width_by
            new_y = patch.get_y() + position_idx*new_height
            #-----
            patch.set_y(new_y)
            patch.set_height(new_height)
    return ax
    
    
#**************************************************
def get_bins_clip_minmax(
    min_max_and_bin_size, 
    include_overflow, 
    include_underflow, 
    allow_max_to_expand_to_fit_int_n_bins=True
):
    r"""
    NOTES ABOUT BINNING----------------------------------------------
    From documentation for pd.hist (appears true for sns.histplot as well)
      The bins argument, when given as a sequence, gives the bin edges,
      INCLUDING left edge of first bin and right edge of last bin.
    Thus, the lower AND upper edges are included.  Due to this, I need to do a bit
      of work to get the bins in the correct form
    --------------------------
    As an example, consider I have a histogram with values ranging from 0 to 15
      and I only want to plot 0-10 (excluding 10)
      Naively, one would end up with bins = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]
      However, as the right edge of last bin is included, the last bin will include values which are
        equal to 9 and those which are equal to 10.  This is not what is desired.
    df = pd.DataFrame({'col_0': [0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 6, 6, 6, 6, 
                                7, 7, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 13, 14]})
    Using df and bins defined above, calling df['col_0'].hist(bins=bins)
      one would find the last bar to contain 6 entries, not the expected number 2
      This is actually the 2 9's plus the 4 10's = 6
    --------------------------
    In order to make this work properly, one needs to add an additional bin to bins.
    I can achieve this by simply repeating the last bin edge twice!
    **************************************************************************************
    Returns a dict containing keys: 'bins', 'clip_min', and 'clip_max'
      TO DRAW THE DESIRED HISTOGRAM:
        df[col].clip(clip_min, clip_max).hist(bins=bins)
                    -- OR --
        sns.histplot(data=df[col].clip(clip_min, clip_max), bins=bins)
    
    
    If allow_max_to_expand_to_fit_int_n_bins==True
      the maximum value in bins will be adjusted such that an integer number of
      bins fits between min_in_plot and max_in_plot
    
    If allow_max_to_expand_to_fit_int_n_bins==False
      the maximum value in bins will not be adjusted.  Typically, the last bin will not
      have the desired bin_size (unless it happens that an integer
      number of bins falls between min_in_plot and max_in_plot)
    
    EXAMPLE (assuming over- and underflow are not included, and forgetting about the
             duplicate last value trick mentioned below)
    min_max_and_bin_size = (0, 10, 1.2)
      If allow_max_to_expand_to_fit_int_n_bins==True
        bins = [ 0. ,  1.2,  2.4,  3.6,  4.8,  6. ,  7.2,  8.4,  9.6, 10.8]
      If allow_max_to_expand_to_fit_int_n_bins==False
        bins = [ 0. ,  1.2,  2.4,  3.6,  4.8,  6. ,  7.2,  8.4,  9.6, 10.]
    """
    #---------------------------------------------
    assert(len(min_max_and_bin_size)==3)
    min_in_plot, max_in_plot, bin_size = min_max_and_bin_size
    #-----
    n_bins = round(np.ceil((max_in_plot-min_in_plot)/bin_size))
    expanded_max_in_plot = min_in_plot + n_bins*bin_size
    # n_bins+1 below because n_bins require n_bins+1 edges to define them
    bins = np.linspace(min_in_plot, expanded_max_in_plot, n_bins+1)
    #-----
    if allow_max_to_expand_to_fit_int_n_bins:
        max_in_plot = expanded_max_in_plot
    else:
        bins[-1] = max_in_plot
    #-----
    # The following (inserting a duplicate last bin) is a trick to make the
    # right edge of the last bin exclusive, like in normal histograms
    bins = np.append(bins, max_in_plot)
    #-----
    # Note: clip will be used to ensure all values <min_in_plot or >max_in_plot
    #       will be counted in the underflow and overflow bins
    # Note: I am seeing somewhat strange behavior in that sys.float_info.epsilon is not exactly the raw machine precision
    #         but maybe more of a relative machine precision.
    #       I believe the problem should be solved by changing, e.g, 
    #           clip_max = max_in_plot+sys.float_info.epsilon
    #       ==> clip_max = max_in_plot*(1+sys.float_info.epsilon)
    # Note: Always want clip_max slightly larger than max_in_plot and clip_min slightly smaller than min_in_plot,
    #       hence the need for the if/else if/else statements (inside of if include_over(under)flow) when setting clip_max/clip_min
    clip_min = -np.inf
    clip_max = np.inf
    if include_overflow:
        bins = bins[:-1]
        bins = np.append(bins, max_in_plot+bin_size)
        if max_in_plot==0:
            clip_max = sys.float_info.epsilon
        elif max_in_plot>0:
            clip_max = max_in_plot*(1+sys.float_info.epsilon)
        else:
            clip_max = max_in_plot*(1-sys.float_info.epsilon)
    if include_underflow:
        bins = np.insert(bins, 0, min_in_plot-bin_size)
        if min_in_plot==0:
            clip_min = -sys.float_info.epsilon
        elif min_in_plot>0:
            clip_min = min_in_plot*(1-sys.float_info.epsilon)
        else:
            clip_min = min_in_plot*(1+sys.float_info.epsilon)
    #-----
    return {'bins':bins, 
            'clip_min':clip_min, 
            'clip_max':clip_max}
            
#**************************************************
def set_box_color_attributes(plot_kwargs, keep_edges_opaque=True):
    r"""
    This adjusts the facecolor, edgecolor, and alpha values
    
    facecolor and edgecolor can be set separately, if desired.
      This allows user to set different colors AND different alpha values
        e.g., facecolor=mcolors.to_rgba('red', 0.25) and edgecolor=mcolors.to_rgba('red', 1.0)
              will make the face transparent but the edges opaque.
              Same functionality can be achieved by setting keep_edges_opaque=True
    If only color is included in hist_plot_kwargs, facecolor and edgecolor are set
      equal to hist_plot_kwargs['color']
    If color and facecolor are included, facecolor is used and color is ignored.
    If edgecolor is not specified:
      If there ARE hatches, edgecolor will be set equal to facecolor
      If there ARE NOT hatches, edgecolor will be set to black
    If keep_edges_opaque==True, the alpha value for the edgecolor will be 1.0
      irrespective of alpha value given in argument
    If alpha is included, this will be used for both facecolor and edgecolor, 
      (unless keep_edges_opaque==True in which case edgecolor alpha will be 1.0),
      i.e., if facecolor=mcolors.to_rgba('red', 0.25) and alpha=1.0, 
               facecolor will become mcolors.to_rgba('red', 1.0)
    NOTE: If palette is set, this will override color and facecolor settings
    -----             
    Fill:
    Set fill=False to make the histogram have no face color (or, make alpha=0)
    """
    #-------------------------
    if plot_kwargs is None:
        plot_kwargs = {}
    #--------------------------------------------------
    #**************************************************
    # If palette is set it overrides color and facecolor
    if plot_kwargs.get('palette', None) is not None:
        plot_kwargs.pop('color', None)
        plot_kwargs.pop('facecolor', None)
        return plot_kwargs
    #**************************************************
    #--------------------------------------------------
    # Fill set to True by default
    # Note: This is typically done by default when calling plotting function (e.g., in pd.hist, sns.histplot, sns.barplot)
    #       This line of code is here mainly to remind me that I can set it to False
    plot_kwargs['fill'] = plot_kwargs.get('fill', True)
    #-------------------------
    # Need to set facecolor and edgecolor
    #-----
    # pop off 'color' and 'alpha' as I don't want these to propagate down to the actual plotting call
    #   color will only be used if facecolor is not included
    #   If alpha is included, it will be used in place of anything already in facecolor/edgecolor
    color = plot_kwargs.pop('color', 'C0') # C0 is the default blue color, if none set
    alpha = plot_kwargs.pop('alpha', None)
    #-----
    plot_kwargs['facecolor'] = plot_kwargs.get('facecolor', None)
    if plot_kwargs['facecolor'] is None:
        plot_kwargs['facecolor'] = color
    #-----
    plot_kwargs['edgecolor'] = plot_kwargs.get('edgecolor', None)
    if plot_kwargs['edgecolor'] is None:
        if plot_kwargs.get('hatch', None) is not None:
            plot_kwargs['edgecolor'] = plot_kwargs['facecolor']
        else:
            plot_kwargs['edgecolor'] = 'black'
    #-----
    # Convert colors to rgba.  If color already in rgba, this doesn't do anything.
    # If rgb color given, a (alpha) set to 1.0
    # Note: After being converted to_rgba, the last value [-1] is alpha
    plot_kwargs['facecolor'] = mcolors.to_rgba(plot_kwargs['facecolor'])
    plot_kwargs['edgecolor'] = mcolors.to_rgba(plot_kwargs['edgecolor'])
    #-----
    if alpha is not None:
        plot_kwargs['facecolor'] = plot_kwargs['facecolor'][:-1] + tuple([alpha])
    #-----    
    if keep_edges_opaque:
        plot_kwargs['edgecolor'] = plot_kwargs['edgecolor'][:-1] + tuple([1.0])
    else:
        # Otherwise, set edgecolor opacity to that of the facecolor
        plot_kwargs['edgecolor'] = plot_kwargs['edgecolor'][:-1] + tuple([plot_kwargs['facecolor'][-1]])
    #-------------------------
    # If facecolor is opaque and edgecolor=facecolor, the hatches will not be visible
    # If this occurs, set the edge colors to black
    if (plot_kwargs.get('hatch', None) is not None and                       # hatches to be drawn
        plot_kwargs['facecolor'][-1]==1 and                                  # facecolor is opaque 
        plot_kwargs['fill']==True and                                        # facecolor is filled
        plot_kwargs['facecolor'][:-1]==plot_kwargs['edgecolor'][:-1]):  # facecolor matches edgecolor
        plot_kwargs['edgecolor'] = 'black'
    #-------------------------
    return plot_kwargs

#**************************************************
def plot_hist(
    ax, 
    df, 
    x_col, 
    min_max_and_bin_size, 
    include_over_underflow=False, 
    stat='count', 
    plot_sns=False, 
    hist_plot_kwargs=None, 
    keep_edges_opaque=True, 
    div_drawn_width_by=None, 
    relative_position_idx=None, 
    run_set_general_plotting_args=True, 
    orient='v', 
    **kwargs
):
    r"""
    Either uses:
       pandas.hist if plot_sns==False
       seaborn.histplot if plot_sns==True
    
    NOTE: Using Seaborn (i.e., plot_sns=True) allows many more parameters to be utilized
          in hist_plot_kwargs
    
    min_max_and_bin_size:
      If min_max_and_bin_size is None, use default number of bins in pandas.DataFrame.hist, which is 10
      If min_max_and_bin_size is an integer, take that the be the number of bins
      Otherwise, min_max_and_bin_size must be a list/tuple of length 3 containing 
        min_in_plot, max_in_plot, and bin_size
    
    include_over_underflow:
      This can be a single boolean, or a list/tuple of two booleans
      If single boolean: include_overflow = include_underflow = include_over_underflow
      If pair of booleans: include_overflow = include_over_underflow[0], 
                           include_underflow = include_over_underflow[1]
    stat:
      Possible values for stat:
        If plot_sns==True
          count: show the number of observations in each bin
          frequency: show the number of observations divided by the bin width
          probability: or proportion: normalize such that bar heights sum to 1
          density: normalize such that the total area of the histogram equals 1
        If plot_sns==False
          ***The pandas.hist method only has a boolean density argument, so it is only
             able to plot the equivalent of count or density above
          count: show the number of observations in each bin
          density: normalize such that the total area of the histogram equals 1
        
    plot_sns:
      dictates whether (True) or not (False) Seaborn is used
    
    hist_plot_kwargs:
      fed to the plotting function, i.e., either pd.hist or sns.histplot
      -----
      Colors:
        facecolor and edgecolor can be set separately, if desired.
          This allows user to set different colors AND different alpha values
            e.g., facecolor=mcolors.to_rgba('red', 0.25) and edgecolor=mcolors.to_rgba('red', 1.0)
                  will make the face transparent but the edges opaque.
                  Same functionality can be achieved by setting keep_edges_opaque=True
        If only color is included in hist_plot_kwargs, facecolor and edgecolor are set
          equal to hist_plot_kwargs['color']
        If color and facecolor are included, facecolor is used and color is ignored.
        If edgecolor is not specified:
          If there ARE hatches, edgecolor will be set equal to facecolor
          If there ARE NOT hatches, edgecolor will be set to black
        If keep_edges_opaque==True, the alpha value for the edgecolor will be 1.0
          irrespective of alpha value given in argument
        If alpha is included, this will be used for both facecolor and edgecolor, 
          (unless keep_edges_opaque==True in which case edgecolor alpha will be 1.0),
          i.e., if facecolor=mcolors.to_rgba('red', 0.25) and alpha=1.0, 
                   facecolor will become mcolors.to_rgba('red', 1.0)
        NOTE: If palette is set, this will override color and facecolor settings
      -----             
      Fill:
        Set fill=False to make the histogram have no face color (or, make alpha=0)
        
    orient:
      Orientation should be 'v' for vertical (default) or 'h' for horizontal.
      NOTE: This overrides any value for orient that may have been set in hist_plot_kwargs
    
    kwargs (SEE Plot_General.set_general_plotting_args for most updated list!):
      ---------- HANDLED BY Plot_General.set_general_plotting_args ----------
      title_args:
        ax.set_title(**title_args)
      ax_args:
        ax.set(**ax_args)
      xlabel_args:
        ax.set_xlabel(**xlabel_args)
      ylabel_args:
        ax.set_ylabel(**ylabel_args)
      draw_legend:
        Draw legend if True (Default = False)
      legend_args:  
        ax.legend(**legend_args)
      tick_args:
        - tick_args can be a dict or a list
        - Making it a list allows operations on both x and y
            e.g. tick_args =[dict(axis='x', labelrotation=90, labelsize=7.0, direction='in'), 
                             dict(axis='y', labelrotation=0, labelsize=10.0, direction='out')]
        for t_args in tick_args:
            ax.tick_params(**t_args)
      grid_args:
        - Default to False, causing grid to be off for both x and y
        - grid_args can be a single boolean, a dict, or a list
        - Making it a list allows operations on both x and y
            e.g. grid_args =[dict(visible=True, axis='x', which='both', color='red'), 
                             dict(visible=True, axis='y', which='major', color='green')]
            - One can also make the elements of the list two booleans, where the first
              will be interpreted as the x-axis on/off and the second the y-axis on/off
                e.g. grid_args=[True,False] will turn x on and y off
              At this point, no mixing and matching bool with dict, as this could lead to 
              unpredictable results
                e.g. grid_args = [dict(visible=True, axis='y', which='major', color='green'), False]
                     would set the y grids on then off, which certainly would not be the desired functionality.
        for g_args in grid_args:
            ax.grid(**g_args)
            
      ---------- NOT HANDLED BY Plot_General.set_general_plotting_args ----------      
      allow_max_to_expand_to_fit_int_n_bins:
        - Input into get_bins_clip_minmax
        
    run_set_general_plotting_args:
      If True, kwargs are fed to Plot_General.set_general_plotting_args
      One would want this set to False when e.g. plotting multiple barplots.
        In such a case, the first plotted should handle the general kwargs, and all
        others should leave them alone.
    """
    #-------------------------
    assert(orient=='v' or orient=='h')
    #-------------------------
    n_patches_beg = len(ax.patches) # Needed in case properties of over/underflow bins need changed
    #---------------------------
    if plot_sns:
        assert(stat in ['count', 'frequency', 'probability', 'density'])
    else:
        assert(stat in ['count', 'density'])
    #---------------------------
    assert(isinstance(include_over_underflow, bool) or 
           isinstance(include_over_underflow, list) or 
           isinstance(include_over_underflow, tuple))
    if isinstance(include_over_underflow, bool):
        include_overflow = include_underflow = include_over_underflow
    else:
        include_overflow = include_over_underflow[0]
        include_underflow = include_over_underflow[1]
    #---------------------------
    allow_max_to_expand_to_fit_int_n_bins = kwargs.get('allow_max_to_expand_to_fit_int_n_bins', True)
    #---------------------------
    hist_plot_kwargs = set_box_color_attributes(
        plot_kwargs=hist_plot_kwargs, 
        keep_edges_opaque=keep_edges_opaque
    )
    #---------------------------
    clip_min = -np.inf
    clip_max = np.inf
    if min_max_and_bin_size is None:
        bins=10 # default value
    elif isinstance(min_max_and_bin_size, int):
        bins = min_max_and_bin_size
    else:
        bins_clip_minmax_dict = get_bins_clip_minmax(min_max_and_bin_size, 
                                                     include_overflow, include_underflow, 
                                                     allow_max_to_expand_to_fit_int_n_bins)
        bins     = bins_clip_minmax_dict['bins']
        clip_min = bins_clip_minmax_dict['clip_min']
        clip_max = bins_clip_minmax_dict['clip_max']            
    #------------------------------------------------------
    #******************************************************
    if plot_sns:
        if orient=='v':
            hist_plot_kwargs['x'] = x_col
        else:
            hist_plot_kwargs['y'] = x_col
        sns.histplot(ax=ax, data=df[[x_col]].clip(clip_min, clip_max), 
                     bins=bins, stat=stat, **hist_plot_kwargs);
    else:
        density = True if stat=='density' else False
        hist_plot_kwargs['orientation'] = 'vertical' if orient=='v' else 'horizontal'
        df[x_col].clip(clip_min, clip_max).hist(ax=ax, bins=bins, density=density, **hist_plot_kwargs);
    #******************************************************
    #------------------------------------------------------
    if run_set_general_plotting_args:
        ax = Plot_General.set_general_plotting_args(ax=ax, **kwargs)
    #---------------------------
    # Give overflow and underflow hatch pattern to distinguish
    n_patches_end = len(ax.patches) # Needed in case properties of over/underflow bins need changed
    if min_max_and_bin_size is not None and (include_overflow or include_underflow):
        # The old method of changing properties of ax.patches[0] and ax.patches[-1] worked if only a 
        #   single histogram is drawn on the plot (and assuming no patches already in created).
        # And, actually, using ax.patches[-1] works for the overflow bins, as the most recent overflow bin
        #   is always the last patch.
        # However, ax.patches[0] will only change the underflow bin of the first histogram even if multiple
        #   histograms are drawn
        # For this reason, and for the more general case of allowing overflow AND/OR underflow as opposed to
        #   just overflow AND underflow, it is necessary to use n_patches_beg and n_patches_end
        if include_underflow:
            # hatches only drawn if fill=True, so need to set to be sure underflow is drawn correctly
            #   even if other bins in histogram not filled
            ax.patches[n_patches_beg].set_fill(True)
            ax.patches[n_patches_beg].set_hatch('*-')
            ax.patches[n_patches_beg].set_edgecolor('white')
        if include_overflow:
            # hatches only drawn if fill=True, so need to set to be sure overflow is drawn correctly
            #   even if other bins in histogram not filled
            ax.patches[n_patches_end-1].set_fill(True)
            ax.patches[n_patches_end-1].set_hatch('*-')
            ax.patches[n_patches_end-1].set_edgecolor('white')
    #---------------------------
    if div_drawn_width_by is not None:
        if relative_position_idx is None:
            relative_position_idx = 0
        ax = adjust_patch_positions_and_widths(
            ax, 
            div_drawn_width_by, 
            relative_position_idx, 
            i_patch_beg=n_patches_beg, 
            i_patch_end=n_patches_end, 
            orient=orient
        )
    #---------------------------
    return ax

#**************************************************
def plot_multiple_hists(
    ax, 
    dfs_w_args, 
    x_col, 
    min_max_and_bin_size, 
    include_over_underflow=False, 
    stat='count', 
    plot_sns=False, 
    keep_edges_opaque=True, 
    include_hatches=False, 
    draw_side_by_side=False, 
    draw_single_idx_full_width=None, 
    orient='v', 
    **kwargs
):
    r"""
    Either uses:
       pandas.hist if plot_sns==False
       seaborn.histplot if plot_sns==True
    
    NOTE: Using Seaborn (i.e., plot_sns=True) allows many more parameters to be utilized
          in hist_plot_kwargs
    
    See plot_hist for more information
    
    dfs_w_args should be a list of:
       i. pd.DataFrame objects
            In this case, the associated barplot_kwargs to be fed into plot_barplot will be an
            empty dict
      ii. lists/tuples of length 2, where the first element is the pd.DataFrame object
          and the second element is a dict representing the associated barplot_kwargs to be
          fed into plot_barplot
    
    include_hatches will add hatches if none are specified in each DFs associated hist_plot_kwargs
    
    If min_max_and_bin_size is None, use default number of bins in pandas.DataFrame.hist, which is 10
    If min_max_and_bin_size is an integer, take that the be the number of bins
    Otherwise, min_max_and_bin_size must be a list/tuple of length 3 containing 
      min_in_plot, max_in_plot, and bin_size
    
    If draw_side_by_side==True, the histograms are drawn side-by-side instead of on top of each other
    If draw_single_idx_full_width is not None, it must be an integer [0, len(dfs_w_args)), and will cause
      the histogram at that index position in dfs_w_args to be drawn at full width.
      This is good if one wants to compare an overall distribution (wide one) to others (narrow ones)
      
    orient:
      Orientation should be 'v' for vertical (default) or 'h' for horizontal.
      NOTE: This overrides any value for orient that may have been set in hist_plot_kwargs in dfs_w_args
    """
    #-------------------------
    # Don't want to alter original dfs_w_args
    dfs_w_args = deepcopy(dfs_w_args)
    #-------------------------
    assert(orient=='v' or orient=='h')
    #-------------------------
    dfs_color_palette = kwargs.get('dfs_color_palette', None) #Only used if colors not specified
    #---------------------------
    if draw_side_by_side:
        div_drawn_width_by = len(dfs_w_args)+1 # +1 so there is empty space between groupings
        if draw_single_idx_full_width is not None:
            assert(draw_single_idx_full_width>=0 and draw_single_idx_full_width<len(dfs_w_args))
            div_drawn_width_by = len(dfs_w_args)-1 # -1 because one of histograms will be wide and behind all others
                                                   # and don't want empty space between
    else:
        div_drawn_width_by=None
    #---------------------------
    colors = Plot_General.get_standard_colors(len(dfs_w_args), palette=dfs_color_palette)
    hatches = Plot_General.get_standard_hatches(len(dfs_w_args))
    for i in range(len(dfs_w_args)):
        # First, normalize dfs_w_args[i] so it is a tuple whose first first element
        # is the pd.DataFrame and second is a dict containing the kwargs
        assert(Utilities.is_object_one_of_types(dfs_w_args[i], [pd.DataFrame, list, tuple]))
        if isinstance(dfs_w_args[i], pd.DataFrame):
            dfs_w_args[i] = (dfs_w_args[i], dict())

        # At this point, dfs_w_args[i] is definitely a list/tuple.  
        # Make sure length=2, first item is pd.DataFrame, and second is dict
        assert(len(dfs_w_args[i])==2 and 
               isinstance(dfs_w_args[i][0], pd.DataFrame) and 
               isinstance(dfs_w_args[i][1], dict))
        
        # However, in order to make changes, it must be a list, and not a tuple!
        if isinstance(dfs_w_args[i], tuple):
            dfs_w_args[i] = list(dfs_w_args[i])

        # Now, set all default values for plot_kwargs_i
        dfs_w_args[i][1]['hatch'] = dfs_w_args[i][1].get('hatch', hatches[i] if include_hatches else None)
        dfs_w_args[i][1]['color'] = dfs_w_args[i][1].get('color', colors[i])
        #dfs_w_args[i][1]['alpha'] = dfs_w_args[i][1].get('alpha', 1.0/len(dfs_w_args))
        dfs_w_args[i][1]['label'] = dfs_w_args[i][1].get('label', f'{i}')
    #---------------------------------------------------------------------
    sbs_count=0
    for i in range(len(dfs_w_args)):
        div_drawn_width_by_i = div_drawn_width_by
        if draw_side_by_side:
            if (draw_single_idx_full_width is not None and 
                draw_single_idx_full_width==i):
                div_drawn_width_by_i=None
                relative_position_idx=None
            else:
                relative_position_idx = sbs_count
                sbs_count += 1
        else:
            relative_position_idx=None
        #----------------------------
        if i==len(dfs_w_args)-1:
            run_set_general_plotting_args=True
        else:
            run_set_general_plotting_args=False
        #----------------------------
        ax = plot_hist(
            ax=ax, 
            df=dfs_w_args[i][0], 
            x_col=x_col, 
            min_max_and_bin_size=min_max_and_bin_size, 
            include_over_underflow=include_over_underflow, 
            stat=stat, 
            plot_sns=plot_sns, 
            hist_plot_kwargs=dfs_w_args[i][1], 
            keep_edges_opaque=keep_edges_opaque, 
            div_drawn_width_by=div_drawn_width_by_i, 
            relative_position_idx=relative_position_idx, 
            run_set_general_plotting_args=run_set_general_plotting_args, 
            orient=orient, 
            **kwargs
        )
            
    return ax