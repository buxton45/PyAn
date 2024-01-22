#!/usr/bin/env python

"""
Utilities specifically for plotting bar plots (using seaborn barplot)
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
import Plot_Hist
#---------------------------------------------------------------------

#**************************************************
def adjust_bar_and_line_positions_and_widths(
    ax, 
    div_width_by, 
    position_idx, 
    i_patch_beg=None, 
    i_patch_end=None, 
    orient='v'
):
    r"""
    This is similar to the Plot_Hist.adjust_patch_positions_and_widths.
    However, with barplots, there is also an error bar associated with each bar which also needs
    its position and width adjusted accordingly.
    
    The documentation below is mainly copied from Plot_Hist.adjust_patch_positions_and_widths.
    
    This is intended to shrink the width of barplot bins and shift them to allow multiple barplots
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
    ax = Plot_Hist.adjust_patch_positions_and_widths(
        ax=ax, 
        div_width_by=div_width_by, 
        position_idx=position_idx, 
        i_patch_beg=i_patch_beg, 
        i_patch_end=i_patch_end, 
        orient=orient
    )
    #-------------------------
    # I suppose there could be an instance where these are not equal?
    # Like maybe if legend or other object drawn before this function is called
    # But, for now, keep this in as safeguard
    assert(len(ax.patches)==len(ax.lines))
    lines   = ax.lines[i_patch_beg:i_patch_end]
    patches = ax.patches[i_patch_beg:i_patch_end]
    # Note: the x-position of the bar is defined as the left edge
    # ==> position of line should be patch_x + 0.5*patch_width
    # Note: Position shifting, according to position_idx, already handled for patches above
    #       in Plot_Hist.adjust_patch_positions_and_widths call.  Therefore, do not need to be
    #       considered below for lines, and can simply use new patch positions.
    for i in range(len(lines)):
        line = lines[i]
        patch = patches[i]
        #----------
        new_line_width = 0.5*line.get_linewidth()
        lines[i].set_linewidth(new_line_width)
        #----------
        if orient=='v':
            new_line_pos = patch.get_x()+0.5*patch.get_width()
            lines[i].set_xdata(new_line_pos)
        else:
            new_line_pos = patch.get_y()+0.5*patch.get_height()
            lines[i].set_ydata(new_line_pos)
    #-------------------------
    return ax

#**************************************************
def plot_barplot(
    ax, 
    df, 
    x=None,
    y=None,
    hue=None, 
    order=None, 
    n_bars_to_include=None, 
    barplot_kwargs=None, 
    keep_edges_opaque=True, 
    div_drawn_width_by=None, 
    relative_position_idx=None, 
    run_set_general_plotting_args=True, 
    orient='v', 
    replace_xtick_labels_with_ints=False, 
    xtick_ints_offset=0, 
    add_xtick_labels_legend_textbox=False, 
    xtick_labels_legend_textbox_kwargs=None, 
    fig=None, 
    **kwargs
):
    r"""
    Uses seaborn sns.barplot
    
    x,y
      x and y variables, treat them as if orientation will be vertical, as they will
      be correctly swapped if horizontal.
    
    orient:
      Orientation should be 'v' for vertical (default) or 'h' for horizontal.
      In terms of the x and y variables, treat them as if plot will be vertical,
        if the orientation is horizontal they will be swapped.
      NOTE: This overrides any value for orient that may have been set in barplot_kwargs
    
    barplot_kwargs:
      fed to the plotting function, sns.barplot
      -----
      Colors:
        facecolor and edgecolor can be set separately, if desired.
          This allows user to set different colors AND different alpha values
            e.g., facecolor=mcolors.to_rgba('red', 0.25) and edgecolor=mcolors.to_rgba('red', 1.0)
                  will make the face transparent but the edges opaque.
                  Same functionality can be achieved by setting keep_edges_opaque=True
        If only color is included in barplot_kwargs, facecolor and edgecolor are set
          equal to barplot_kwargs['color']
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
    
    ----------
    replace_xtick_labels_with_ints:
      If True, the xtick labels will be replaced with integers, and a key will be printed to the right of the figures.
      
    xtick_ints_offset:
      Set to nonzero value if one wants the counting to start at a value other than 1
      
    add_xtick_labels_legend_textbox:
      Only has effect if replace_xtick_labels_with_ints==True.
      NOTE: If add_xtick_labels_legend_textbox==True, fig MUST BE PROVIDED!
      If add_xtick_labels_legend_textbox==True, Plot_General.generate_xtick_labels_legend_textbox is called and a textbox is generated
        with a legend to translate the xtick_label ints to their true values.
        
    xtick_labels_legend_textbox_kwargs:
      Additional arguments for Plot_General.generate_xtick_labels_legend_textbox
        
    fig:
      Reference to the figure within which the plot is generated.
      NOTE: NEEDED WHEN replace_xtick_labels_with_ints==add_xtick_labels_legend_textbox==True, as the fig reference is needed to draw
            the text box.
    ----------
    
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
      ---------- HANDLED BY Plot_General.set_general_plotting_args ----------
      N/A
            
    run_set_general_plotting_args:
      If True, kwargs are fed to Plot_General.set_general_plotting_args
      One would want this set to False when e.g. plotting multiple barplots.
        In such a case, the first plotted should handle the general kwargs, and all
        others should leave them alone.
    """
    #-------------------------
    if replace_xtick_labels_with_ints and add_xtick_labels_legend_textbox:
        assert(fig is not None)
    #-------------------------
    assert(orient=='v' or orient=='h')
    if orient=='h':
        x,y = y,x
    #-------------------------
    #---------------------------
    n_patches_beg = len(ax.patches) # Needed in case adjust_bar_and_line_positions_and_widths used
    #---------------------------
    barplot_kwargs = Plot_Hist.set_box_color_attributes(
        plot_kwargs=barplot_kwargs, 
        keep_edges_opaque=keep_edges_opaque
    )
    barplot_kwargs['orient'] = orient
    #---------------------------
    if order is not None:
        if n_bars_to_include is not None:
            order=order[:n_bars_to_include]
        df_to_plot = df[order]
    else:
        df_to_plot = df
    #-----
    if n_bars_to_include is not None:
        df_to_plot = df_to_plot.iloc[:, :n_bars_to_include]
    #------------------------------------------------------
    #******************************************************
    sns.barplot(ax=ax, data=df_to_plot, x=x, y=y, hue=hue, order=order, **barplot_kwargs);
    #******************************************************
    #------------------------------------------------------
    if run_set_general_plotting_args:
        ax = Plot_General.set_general_plotting_args(ax=ax, **kwargs)
    #---------------------------
    n_patches_end = len(ax.patches) # Needed in case properties of over/underflow bins need changed
    #---------------------------
    if div_drawn_width_by is not None:
        if relative_position_idx is None:
            relative_position_idx = 0
        ax = adjust_bar_and_line_positions_and_widths(
            ax=ax, 
            div_width_by=div_drawn_width_by, 
            position_idx=relative_position_idx, 
            i_patch_beg=n_patches_beg, 
            i_patch_end=n_patches_end, 
            orient=orient
        )
    #---------------------------
    # Regardless of run_set_general_plotting_args, if replace_xtick_labels_with_ints==True
    #   Plot_General.set_general_plotting_args needs to be called with new tick elements
    if replace_xtick_labels_with_ints:
        xtick_elements = df_to_plot.columns.tolist()
        xtick_rename_dict = {xtick_el:i+1+xtick_ints_offset for i,xtick_el in enumerate(xtick_elements)}
        # NOTE: xticks = np.arange(len(xtick_rename_dict)) below is to ensure all ticks are drawn,
        #       as sometimes mpl draws less ticks when there are many
        ax = Plot_General.set_general_plotting_args(
            ax=ax, 
            ax_args=dict(
                xticks = np.arange(len(xtick_rename_dict)),
                xticklabels=list(xtick_rename_dict.values())
            ), 
            tick_args=dict(axis='x', labelrotation=0)
        )
        if add_xtick_labels_legend_textbox:
            subplot_layout_params = Plot_General.get_subplot_layout_params(fig)
            dflt_xtick_labels_legend_textbox_kwargs = dict(
                fig=fig, 
                xtick_rename_dict=xtick_rename_dict, 
                text_x_pos=1.02*subplot_layout_params['right'], 
                text_y_pos=subplot_layout_params['top'], 
                n_chars_per_line=30, 
                multi_line_offset=None, 
                new_org_separator=': ', 
                fontsize=18, 
                ha='left', 
                va='top',
                n_lines_between_entries=1, 
                n_cols=1, 
                col_padding = 0.01
            )
            if xtick_labels_legend_textbox_kwargs is None:
                xtick_labels_legend_textbox_kwargs={}
            xtick_labels_legend_textbox_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict=xtick_labels_legend_textbox_kwargs, 
                default_values_dict=dflt_xtick_labels_legend_textbox_kwargs
            )
            Plot_General.generate_xtick_labels_legend_textbox(
                **xtick_labels_legend_textbox_kwargs
            )
    #---------------------------
    return ax
    
    
#**************************************************
def plot_multiple_barplots(
    ax, 
    dfs_w_args, 
    x=None,
    y=None,
    hue=None, 
    order=None, 
    n_bars_to_include=None, 
    keep_edges_opaque=True, 
    include_hatches=False, 
    draw_side_by_side=False, 
    draw_single_idx_full_width=None,     
    run_set_general_plotting_args=True, 
    orient='v', 
    replace_xtick_labels_with_ints=False,
    xtick_ints_offset=0, 
    add_xtick_labels_legend_textbox=False, 
    xtick_labels_legend_textbox_kwargs=None, 
    fig=None, 
    **kwargs
):
    r"""
    Uses seaborn barplot
    
    For plots to be drawn together, it is important that they all have the same ordering.
    Therefore, if order is None, the column order of the first df in dfs_w_args will be used for ordering.
    
    See plot_barplot for more information
    
    dfs_w_args should be a list of:
       i. pd.DataFrame objects
            In this case, the associated barplot_kwargs to be fed into plot_barplot will be an
            empty dict
      ii. lists/tuples of length 2, where the first element is the pd.DataFrame object
          and the second element is a dict representing the associated barplot_kwargs to be
          fed into plot_barplot
          
    x,y
      x and y variables, treat them as if orientation will be vertical, as they will
      be correctly swapped if horizontal.
    
    include_hatches will add hatches if none are specified in each DFs associated barplot_kwargs
    
    If draw_side_by_side==True, the bar plots are drawn side-by-side instead of on top of each other
    If draw_single_idx_full_width is not None, it must be an integer [0, len(dfs_w_args)), and will cause
      the bar plot at that index position in dfs_w_args to be drawn at full width.
      This is good if one wants to compare an overall distribution (wide one) to others (narrow ones)
      
    orient:
      Orientation should be 'v' for vertical (default) or 'h' for horizontal.
      In terms of the x and y variables, treat them as if plot will be vertical,
        if the orientation is horizontal they will be swapped.
      NOTE: This overrides any value for orient that may have been set in barplot_kwargs in dfs_w_args
      
    ----------
    replace_xtick_labels_with_ints:
      If True, the xtick labels will be replaced with integers, and a key will be printed to the right of the figures.
      
    xtick_ints_offset:
      Set to nonzero value if one wants the counting to start at a value other than 1
      
    add_xtick_labels_legend_textbox:
      Only has effect if replace_xtick_labels_with_ints==True.
      NOTE: If add_xtick_labels_legend_textbox==True, fig MUST BE PROVIDED!
      If add_xtick_labels_legend_textbox==True, Plot_General.generate_xtick_labels_legend_textbox is called and a textbox is generated
        with a legend to translate the xtick_label ints to their true values.
        
    xtick_labels_legend_textbox_kwargs:
      Additional arguments for Plot_General.generate_xtick_labels_legend_textbox
        
    fig:
      Reference to the figure within which the plot is generated.
      NOTE: NEEDED WHEN replace_xtick_labels_with_ints==add_xtick_labels_legend_textbox==True, as the fig reference is needed to draw
            the text box.
    ----------
    """
    #-------------------------
    # Don't want to alter original dfs_w_args
    dfs_w_args = deepcopy(dfs_w_args)
    #-------------------------
    if replace_xtick_labels_with_ints and add_xtick_labels_legend_textbox:
        assert(fig is not None)
    #---------------------------
    dfs_color_palette = kwargs.get('dfs_color_palette', None) #Only used if colors not specified
    #---------------------------
    if draw_side_by_side:
        # NOTE: barplots already have space between groupings, so no need to add or subtract
        # 1 from len(dfs_w_args) here (as is done in Plot_Hist.plot_multiple_hists)
        div_drawn_width_by = len(dfs_w_args)
        if draw_single_idx_full_width is not None:
            assert(draw_single_idx_full_width>=0 and draw_single_idx_full_width<len(dfs_w_args))
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
    # Make sure ordering and columns all equal!
    if order is None:
        order = dfs_w_args[0][0].columns.tolist()
    if n_bars_to_include is not None:
        order=order[:n_bars_to_include]
    for i in range(len(dfs_w_args)):
        dfs_w_args[i][0] = dfs_w_args[i][0][order]
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
        # NOTE: order and n_bars_to_include already handled above
        #       so can be set to None below
        # NOTE: Only want xtick_labels_legend_textbox to be drawn once, so
        #       set to False below, and handled at end of function
        ax = plot_barplot(
            ax=ax, 
            df=dfs_w_args[i][0], 
            x=x,
            y=y,
            hue=hue, 
            order=None, 
            n_bars_to_include=None, 
            barplot_kwargs=dfs_w_args[i][1], 
            keep_edges_opaque=keep_edges_opaque, 
            div_drawn_width_by=div_drawn_width_by_i, 
            relative_position_idx=relative_position_idx, 
            run_set_general_plotting_args=run_set_general_plotting_args, 
            orient=orient, 
            replace_xtick_labels_with_ints=replace_xtick_labels_with_ints, 
            xtick_ints_offset=xtick_ints_offset, 
            add_xtick_labels_legend_textbox=False, 
            xtick_labels_legend_textbox_kwargs=None, 
            fig=None, 
            **kwargs
        )
    #---------------------------------------------------------------------
    if replace_xtick_labels_with_ints and add_xtick_labels_legend_textbox:
        xtick_elements = dfs_w_args[0][0].columns.tolist()
        xtick_rename_dict = {xtick_el:i+1+xtick_ints_offset for i,xtick_el in enumerate(xtick_elements)}
        subplot_layout_params = Plot_General.get_subplot_layout_params(fig)
        dflt_xtick_labels_legend_textbox_kwargs = dict(
            fig=fig, 
            xtick_rename_dict=xtick_rename_dict, 
            text_x_pos=1.02*subplot_layout_params['right'], 
            text_y_pos=subplot_layout_params['top'], 
            n_chars_per_line=30, 
            multi_line_offset=None, 
            new_org_separator=': ', 
            fontsize=18, 
            ha='left', 
            va='top',
            n_lines_between_entries=1, 
            n_cols=1, 
            col_padding = 0.01
        )
        if xtick_labels_legend_textbox_kwargs is None:
            xtick_labels_legend_textbox_kwargs={}
        xtick_labels_legend_textbox_kwargs = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict=xtick_labels_legend_textbox_kwargs, 
            default_values_dict=dflt_xtick_labels_legend_textbox_kwargs
        )
        Plot_General.generate_xtick_labels_legend_textbox(
            **xtick_labels_legend_textbox_kwargs
        )    
    #----------------------------
    return ax