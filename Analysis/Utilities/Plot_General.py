#!/usr/bin/env python

"""
Utilities for general plotting
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
import copy
import warnings
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
#---------------------------------------------------------------------

#**************************************************
def save_fig(
    fig, save_dir, save_name, 
    dpi='figure', format=None, metadata=None, 
    bbox_inches=None, pad_inches=0.1,
    facecolor='auto', edgecolor='auto',
    backend=None, **kwargs
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    fig.savefig(
        save_path, 
        dpi=dpi, format=format, metadata=metadata, 
        bbox_inches=bbox_inches, pad_inches=pad_inches,
        facecolor=facecolor, edgecolor=edgecolor,
        backend=backend, **kwargs        
    )
    
#**************************************************
def set_general_plotting_args(ax, **kwargs):
    r"""
    kwargs:
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
    """
    #-------------------------
    title_args  = kwargs.get('title_args', None)
    ax_args     = kwargs.get('ax_args', None)
    xlabel_args = kwargs.get('xlabel_args', None)
    ylabel_args = kwargs.get('ylabel_args', None)
    draw_legend = kwargs.get('draw_legend', None)
    legend_args = kwargs.get('legend_args', None)
    tick_args   = kwargs.get('tick_args', None)
    grid_args   = kwargs.get('grid_args', None)
    #-------------------------
    if isinstance(title_args, str):
        title_args = dict(label=title_args)
    if title_args is not None:
        ax.set_title(**title_args);
    #-------------------------
    if ax_args is not None:
        ax.set(**ax_args);
    if xlabel_args is not None:
        ax.set_xlabel(**xlabel_args);
    if ylabel_args is not None:
        ax.set_ylabel(**ylabel_args);
    #-------------------------
    if tick_args is not None:
        assert(Utilities.is_object_one_of_types(tick_args, [dict, list, tuple]))
        if isinstance(tick_args, dict):
            tick_args = [tick_args]
        assert(len(tick_args)<=2)
        for t_args in tick_args:
            assert(isinstance(t_args, dict))
            ax.tick_params(**t_args);
    #-------------------------
    if grid_args is not None:
        assert(Utilities.is_object_one_of_types(grid_args, [bool, dict, list, tuple]))
        if isinstance(grid_args, bool):
            grid_args = [dict(visible=False, axis='both')]
        if isinstance(grid_args, dict):
            grid_args = [grid_args]
        #-------------------------
        # Now, grid_args should be a list or tuple
        assert(Utilities.is_object_one_of_types(grid_args, [list, tuple]))
        assert(len(grid_args)<=2)
        if(len(grid_args)==2 and 
           isinstance(grid_args[0], bool) and 
           isinstance(grid_args[1], bool)):
            grid_args = [dict(visible=grid_args[0], axis='x'), 
                         dict(visible=grid_args[1], axis='y')]
        for g_args in grid_args:
            assert(isinstance(g_args, dict))
            ax.grid(**g_args);
    #-------------------------
    if draw_legend is not None:
        if draw_legend:
            if legend_args is None:
                ax.legend();
            else:
                ax.legend(**legend_args);
        else:
            # If legend already drawn, remove it
            if ax.get_legend():
                ax.get_legend().remove()
    #-------------------------
    return ax

#**************************************************
#--------------------------------------------------------------------
def get_standard_markers(n_markers, exclude_not_filled_markers=True):
    # This will first grab the set of (35 for all, 23 if exclude_not_filled_markers) markers from Line2D.markers
    # If n_markers is less than the number of unique markers, the list will be chopped down
    # If n_markers is more than the number of unique markers, the list will be repeated
    #   until the list has as many as are needed
    
    #Note, in Line2D.markers.keys(), the first two are "." (point, which looks like a smaller "o")
    #  and "," (pixel)
    #  The last four are 'None', None, ' ', ''
    markers = list(Line2D.markers.keys())[2:-4]
    if exclude_not_filled_markers:
        not_filled_markers = ['1', '2', '3', '4', '+', 'x', '|', '_', 0, 1, 2, 3]
        markers = [x for x in markers if x not in not_filled_markers]        
    
    if n_markers > len(markers):
        # Repeat markers until have as many as n_markers
        tmp_count = 0
        n_unq_mark = len(markers)
        while n_markers > len(markers):
            markers.append(markers[tmp_count % n_unq_mark])
            tmp_count += 1
    else:
        markers = markers[:n_markers]
    return markers


def get_standard_linestyles(n_lines, exclude_loose=False):
    # Similar to get_standard_markers
    # Using the set of linestyle_tuple defined in this function (can easily be expanded if desired),
    #  this will grab a list of n_lines styles
    # If n_lines is less than the number of unique styles, the list will be chopped down
    # If n_lines is more than the number of unique styles, the list will be repeated
    #   until the list has as many as are needed
    #--------------------------------------------------------
    linestyle_tuple = [
        ('solid',                (0, ())),

        ('dotted',                (0, (1, 2))),
        ('dashed',                (0, (4, 4))),
        ('dashdotted',            (0, (4, 2, 1, 2))),
        ('dashdotdotted',         (0, (3, 4, 1, 4, 1, 4))),

        ('densely dotted',        (0, (1, 1))),
        ('densely dashed',        (0, (4, 1))),
        ('densely dashdotted',    (0, (4, 1, 1, 1))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

        ('loosely dotted',        (0, (1, 8))),
        ('loosely dashed',        (0, (4, 8))),
        ('loosely dashdotted',    (0, (4, 8, 1, 8))),
        ('loosely dashdotdotted', (0, (3, 8, 1, 8, 1, 8)))]
        
    if exclude_loose:
        linestyle_tuple = linestyle_tuple[:-4]
    #--------------------------------------------------------
    linestyles = [x[1] for x in linestyle_tuple]
    if n_lines > len(linestyles):
        # Repeat linestyles until have as many as n_lines
        tmp_count = 0
        n_unq_lines = len(linestyles)
        while n_lines > len(linestyles):
            linestyles.append(linestyles[tmp_count % n_unq_lines])
            tmp_count += 1
    else:
        linestyles = linestyles[:n_lines]
    return linestyles
    
    
def get_standard_colors(n_colors, palette=None):
    # colorblind has only 10 different colors
    # If palette is None:
    #   Use colorblind if n_colors < 10
    #   Otherwise use husl (maybe should be Spectral?);
    # Otherwise:
    #   Use palette
    if palette is None:
        if n_colors <= 10:
            colors = sns.color_palette('colorblind', n_colors)
        else:
            colors = sns.color_palette('husl', n_colors)
    else:
        colors = sns.color_palette(palette, n_colors)
    return colors

def get_standard_colors_dict(keys, palette=None):
    n_colors = len(keys)
    palette_dict = get_standard_colors(n_colors, palette)
    palette_dict = {keys[i]:palette_dict[i] for i in range(n_colors)}
    return palette_dict
    
    
def get_standard_hatches(n_hatch, exclude_vertical=True):
    # If n_hatch is less than the number of unique hatches, the list will be chopped down
    # If n_hatch is more than the number of unique hatches, the list will be repeated
    #   until the list has as many as are needed
    # exclude_vertical is an option because for thin bins, a vertical hatching can be confusing
    #   as there will sometime appear to be a random single vertical line in the bin
    
    hatches = ['/', '\\', '|', '-', 'x', '+', 'o', 'O', '.', '*', 
               '//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**', 
               '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.']
    if exclude_vertical:
        hatches = [x for x in hatches if x not in ('|', '||')]
    
    if n_hatch > len(hatches):
        # Repeat hatches until have as many as n_hatch
        tmp_count = 0
        n_unq_hatch = len(hatches)
        while n_hatch > len(hatches):
            hatches.append(hatches[tmp_count % n_unq_hatch])
            tmp_count += 1
    else:
        hatches = hatches[:n_hatch]
    return hatches
    
#--------------------------------------------------------------------


#**************************************************
def get_flattened_axes(
    axs, 
    row_major=True
):
    r"""
    Convert a multi-dimensional axs array to a single dimension.
    This is helpful when iterating over plots, etc.
    
    row_major:
      If True, the array is flattened in row-major order.
      If False, the array is flattened in column-major order.
      
        For the following, assume A = | a11  a12  a13 |
                                      | a21  a22  a23 |
        row_major=True:
          Consecutive elements of a row reside next to each other
            e.g., A.flatten(order='C')==A.flatten(order='C')==[a11, a12, a13, a21, a22, a23]

        row_major=False:
          Consecutive elements of a column reside next to each other
              e.g., A.flatten(order='F')==[a11, a21, a12, a22, a13, a23]
    """
    #-------------------------
    if not isinstance(axs, np.ndarray):
        axs = np.asarray(axs)
    #-------------------------
    if row_major:
        return axs.flatten(order='C') # 'C' for C-style
    else:
        return axs.flatten(order='F') # 'F' for Fortran-style

def default_subplots(
    n_x=1,
    n_y=1,
    fig_num=0,
    sharex=False,
    sharey=False,
    unit_figsize_width=14,
    unit_figsize_height=6, 
    return_flattened_axes=False,
    row_major=True
):
    r"""
    Returns the same as matplotib.pyplot.subplots.
      fig: Figure
      axs: single axes.Axes or array of Axes (if more than one subplot was created).
    ----------------------------------------------------------------------------------------------------  
    A few useful notes on the ordering of axs when a two-dimensional array is created:
        For 2-dimensional plot arrays, the first index denotes the row and the second the column
          This is how I typically think of two-dimensional arrays
        For the following, assume A = | a11  a12  a13 |
                                      | a21  a22  a23 |
        numpy.flatten:
          order='C' (for C-style)
            the default ordering, which is row-major order.  Consecutive elements of a row reside next to each other
              e.g., A.flatten(order='C')==A.flatten(order='C')==[a11, a12, a13, a21, a22, a23]

          order='F' (for Fortran-style)
            flatten in column-major order.  Consecutive elements of a column reside next to each other
              e.g., A.flatten(order='F')==[a11, a21, a12, a22, a13, a23]
    ----------------------------------------------------------------------------------------------------
    n_x:
      Number of columns
      
    n_y:
      Number of rows
      
    fig_num:
      Figure number
      
    sharex, sharey: bool or {'none', 'all', 'row', 'col'}, default: False
      Controls sharing of properties among x (sharex) or y (sharey) axes:
        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.
      When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created. 
      When subplots have a shared y-axis along a row, only the y tick labels of the first column subplot are created.
      
    unit_figsize_width, unit_figsize_height:
      The width and height of a 1x1 figure.
      The figure created here will be scaled as figsize=[unit_figsize_width*n_x, 
                                                         unit_figsize_height*n_y]
                                                         
    return_flattened_axes, row_major:
      If return_flattened_axes=True, flatten axs down to one dimension via get_flattened_axes(axs=axs, row_major=row_major)
    """
    #-------------------------
    if unit_figsize_width is None:
        unit_figsize_width = plt.rcParams["figure.figsize"][0]
    if unit_figsize_height is None:
        unit_figsize_height = plt.rcParams["figure.figsize"][1]
    #-------------------------
    fig, axs = plt.subplots(
        nrows=n_y,
        ncols=n_x,
        num=fig_num, 
        figsize = [
            unit_figsize_width*n_x, 
            unit_figsize_height*n_y
        ], 
        sharex=sharex, 
        sharey=sharey
    )
    #-------------------------
    if return_flattened_axes:
        axs = get_flattened_axes(axs=axs, row_major=row_major)
    #-------------------------
    return fig, axs
    

#**************************************************    
def make_all_axes_have_same_ylims(axs):
    r"""
    Make all of the axes in axs have the same y-limits, taken to be the minimum and maximum
    values for the group
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray(axs)
    y_min = np.min(([ax_i.get_ylim() for ax_i in axs.flatten()]))
    y_max = np.max(([ax_i.get_ylim() for ax_i in axs.flatten()]))
    for ax_i in axs.flatten():
        ax_i.set_ylim(y_min, y_max)
        
    
#**************************************************
def generate_xtick_labels_legend_textbox(
    fig, 
    xtick_rename_dict, 
    text_x_pos,
    text_y_pos, 
    n_chars_per_line=30, 
    multi_line_offset=None, 
    new_org_separator=': ', 
    fontsize=18, 
    ha='left', 
    va='top',
    n_lines_between_entries=1, 
    n_cols=1, 
    col_padding = 0.01
):
    r"""
    Generate a textbox legend for the xtick labels.
    This is useful when the xtick labels are very long, and the user wants to replace
      them with e.g., ints instead.  This will create a textbox to allow the viewer to understand
      what the e.g., ints stand for.
    
    xtick_rename_dict:
      A dict object with keys equal to the original xtick labels and values equal to
      the new xtick labels (i.e., the xtick labels appearing in the plot)
      NOTE: All new xtick labels MUST HAVE LENGTHS SHORTER THAN n_chars_per_line!
      
    text_x_pos, text_x_pos:
      x and y position to insert textbox
      
    n_chars_per_line:
      Number of characters per line.  The text will be split into mutliple lines according to this.
      
    multi_line_offset:
      The leading whitespace (or whatever string one wants, I suppose?) for the non-first line in multi-line 
      entries.  If this is left equal to None, it will be set to empty spaces with length equal to the new_xtick 
      plus the separtor.
      
    new_org_separator:
      Separator between the new and original xtick label entries.
      
    fontsize:
      Font size to be used in textbox
      
    ha:
      Horizontal alignment
      
    va:
      Vertical alignment
      
    n_lines_between_entries:
      Number of lines between entries
      
    n_cols:
      Number of columns.
      If n_cols>1, these are split by the number of elements, not the total length of each column
        So, a column with lots of long, multi-line, entries will be longer than the others
        
    col_padding:
      Spacing between columns
    """
    #----------------------------------------------------------------------------------------------------
    if n_cols>1:
        n_entries_per_column = np.ceil(len(xtick_rename_dict)/n_cols).astype(int)
        for i_col in range(n_cols):
            xtick_keys_in_cols_i = list(xtick_rename_dict.keys())[i_col*n_entries_per_column:(i_col+1)*n_entries_per_column]
            xtick_rename_dict_i = {k:v for k,v in xtick_rename_dict.items() if k in xtick_keys_in_cols_i}
            #----------
            if i_col==0:
                text_x_pos_i=text_x_pos
            else:
                # If i_col>0, use the bounds of the previous text box to place the next box
                #   More specifically, the maximum x position of the previous box is used
                prev_txt = fig.texts[-1]
                # For whatever reason, a renderer is needed for get_window_extent below
                prev_txt_x_max = prev_txt.get_transform().inverted().transform_bbox(
                    prev_txt.get_window_extent(
                        renderer=plt.gcf().canvas.get_renderer()
                    )
                ).x1
                text_x_pos_i=prev_txt_x_max+col_padding
            #----------
            text_y_pos_i=text_y_pos
            #----------
            # NOTE: n_cols must be equal to 1 below, otherwise infinite loop!
            generate_xtick_labels_legend_textbox(
                fig=fig, 
                xtick_rename_dict=xtick_rename_dict_i, 
                text_x_pos=text_x_pos_i,
                text_y_pos=text_y_pos_i, 
                n_chars_per_line=n_chars_per_line, 
                multi_line_offset=multi_line_offset, 
                new_org_separator=new_org_separator, 
                fontsize=fontsize, 
                ha=ha, 
                va=va,
                n_lines_between_entries=n_lines_between_entries, 
                n_cols=1, 
                col_padding=0
            )
        return
    #----------------------------------------------------------------------------------------------------
    text_str = ''
    for org_xtick, new_xtick in xtick_rename_dict.items():
        # If new_xtick is a not a string (typically is an int), make it one
        if not isinstance(new_xtick, str):
            new_xtick = str(new_xtick)
        #-------------------------
        # org_xtick is typically a string, but can also be a tuple (or list I suppose) when the DF being
        #   plotted has MultiIndex columns.  If tuple, convert to string
        assert(Utilities.is_object_one_of_types(org_xtick, [str, tuple, list]))
        if Utilities.is_object_one_of_types(org_xtick, [tuple, list]):
            org_xtick = ', '.join(org_xtick)
        #-------------------------
        # This only really works if new_xtick fits on a single line
        assert(len(new_xtick)<n_chars_per_line)
        #-------------------------
        # If multi_line_offset is not input, set it equal to empty spaces with length equal
        # to the new_xtick plus the separtor
        if multi_line_offset is None:
            multi_line_offset = ' '*len(f'{new_xtick}{new_org_separator}')
        #-------------------------
        # Calculate n_lines
        #-----
        # Estimate of total characters
        total_characters = len(org_xtick)+len(new_xtick)
        n_lines = int(np.ceil(total_characters/n_chars_per_line))
        #-----
        # Calculation of total characters
        total_characters += len(new_org_separator)+(n_lines-1)*len(multi_line_offset)
        n_lines = int(np.ceil(total_characters/n_chars_per_line))
        #-------------------------
        org_xtick_start_pos=0
        for i_line in range(n_lines):
            #-----
            if i_line==0:
                line_beg = f'{new_xtick}{new_org_separator}'
            else:
                line_beg = multi_line_offset
            text_str += line_beg
            remaining_chars_in_line = n_chars_per_line-len(line_beg)
            #-----
            # NOTE: Extending upper limit of slice above total number is fine, it will stop at end of array
            text_str += '{}{}\n'.format(org_xtick[org_xtick_start_pos:org_xtick_start_pos+remaining_chars_in_line].strip(), 
                                                '-' if i_line<n_lines-1 else '')
            org_xtick_start_pos+=remaining_chars_in_line
        text_str += n_lines_between_entries*'\n'
    #-------------------------
    fig.text(x=text_x_pos, y=text_y_pos, s=text_str, fontsize=fontsize, ha=ha, va=va);
    return # Need this return, otherwise figure is drawn twice!

#**************************************************
def remove_duplicates_from_legend(
    ax, 
    ax_props=None
):
    r"""
    Very simple minded function intended for a single legend associated with axis ax.
    If multiple entries exist with the same label, keep only the first
    """
    #-------------------------
    # Want to maintain any settings already set in ax.legend()
    props_to_copy = [
        'alignment',
        'alpha',
        'label',
        'title'
    ]
    dflt_ax_props = {k:v for k,v in ax.legend().properties().items() 
                     if k in props_to_copy}
    dflt_ax_props['title'] = dflt_ax_props['title'].get_label()
    #-----
    if ax_props is None:
        ax_props = dflt_ax_props
    else:
        ax_props = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = ax_props, 
            default_values_dict = dflt_ax_props, 
            extend_any_lists    = False, 
            inplace             = False
        )

    #-------------------------
    handles,labels = ax.get_legend_handles_labels()
    assert(len(labels)==len(handles))
    #-----
    fnl_labels_handles = dict()
    for i in range(len(labels)):
        if labels[i] in fnl_labels_handles.keys():
            continue
        fnl_labels_handles[labels[i]] = handles[i]
    # Safest to convert to list of tuples, so correct pairs are maintained when being set below
    fnl_labels_handles = list(fnl_labels_handles.items())
    #-------------------------
    ax.legend(
        handles = [x[1] for x in fnl_labels_handles], 
        labels  = [x[0] for x in fnl_labels_handles]
    ).set(**ax_props);
    return ax
       
       
#**************************************************           
def adjust_kwargs(
    general_kwargs, 
    new_values_dict, 
    append_to_containers=True, 
    inplace=False
):
    r"""
    Adjusts values of input_dict according to new_values_dict and returns copy.
    Default behavior (inplace=False) is preserve input_dict and return altered version.
    
    append_to_containers:
        If False, values will be replaced
        If True and key exists in general_kwargs:
          If new_values_dict[key] or general_kwargs[key] is a list:
            Extend general_kwargs[key] with the values from new_values_dict[key]
          If new_values_dict[key] and general_kwargs[key] are dicts:
            Replace common keys with new_values_dict[key] values, add non-common keys from new_values_dict[key],
              keep non-common keys from general_kwargs[key]
            NOTE: This only makes sense if both new_values_dict[key] and general_kwargs[key] are dict objects
        If True and key does not exists in general_kwargs:
          Normal functioning, i.e., new_values_dict[key] will be added to general_kwargs
    """
    #-------------------------
    if not inplace:
        general_kwargs = copy.deepcopy(general_kwargs)
    #-------------------------
    for key in new_values_dict.keys():
        if key not in general_kwargs:
            # If the key is not in general_kwargs, add it
            general_kwargs[key] = new_values_dict[key]
        else:
            # If the key is in general_kwargs, the behvaior depends on the append_to_containers setting
            # and the data type of general_kwargs[key] and new_values_dict[key]
            if not append_to_containers:
                # If append_to_containers is False, replace general_kwargs[key] with 
                # new_values_dict[key] in all instances.
                general_kwargs[key] = new_values_dict[key]
            else:
                # If append_to_containers is True: 
                #   - append to general_kwargs[key] if new_values_dict[key] or general_kwargs[key] is a list
                #   - if dict, replace common keys with new_values_dict[key] values, add non-common keys from new_values_dict[key],
                #     keep non-common keys from general_kwargs[key]
                #-------------------------
                if(not Utilities.is_object_one_of_types(general_kwargs[key], [list,dict]) and 
                   not Utilities.is_object_one_of_types(new_values_dict[key], [list,dict])):
                    # If not lists or dicts, replace as normal
                    general_kwargs[key] = new_values_dict[key]
                #-------------------------
                elif(isinstance(general_kwargs[key], list) or isinstance(new_values_dict[key], list)):
                    #-----
                    # Make sure both are lists
                    if not isinstance(general_kwargs[key], list):
                        general_kwargs[key] = [general_kwargs[key]]
                    if not isinstance(new_values_dict[key], list):
                        new_values_dict[key] = [new_values_dict[key]]
                    #-----
                    general_kwargs[key].extend([x for x in new_values_dict[key] 
                                                if x not in general_kwargs[key]])
                #-------------------------
                elif(isinstance(general_kwargs[key], dict) or isinstance(new_values_dict[key], dict)):
                    # Could have simply used and above instead of or (together with the assert(0) in the else below), 
                    # but this will make it easier to understand why a failure occurs.
                    assert(isinstance(general_kwargs[key], dict) and isinstance(new_values_dict[key], dict))
                    for sub_key in new_values_dict[key].keys():
                        general_kwargs[key][sub_key] = new_values_dict[key][sub_key]
                #-------------------------
                else:
                    assert(0)
    #-------------------------
    return general_kwargs
        
#**************************************************
def get_subplots_adjust_args(
    left=None, 
    right=None, 
    bottom=None, 
    top=None, 
    wspace=None, 
    hspace=None,
    
    scale_margin_left=None, 
    scale_margin_right=None, 
    scale_margin_bottom=None, 
    scale_margin_top=None, 
    scale_wspace=None, 
    scale_hspace=None    
):
    #-------------------------
    r"""
    Values are set to default values (rcParams["figure.subplot.[name]"]) if not specified
    left, right, bottom, top:
      self-explanatory
    wspace: 
      The width of the padding between subplots, as a fraction of the average Axes width.
    hspace: 
      The height of the padding between subplots, as a fraction of the average Axes height.
      
    scale_x:
      Scale the corresponding parameters.
      This is intended as a lazy way of adjusting the default parameters without setting absolute values.
        e.g., if one wants the left margin to be 25% larger, call: get_subplots_adjust_args(scale_margin_left=1.25)  
      
    At the time of writing this document, the default values are:
        mpl.rcParams["figure.subplot.left"]   = 0.125
        mpl.rcParams["figure.subplot.right"]  = 0.9
        mpl.rcParams["figure.subplot.bottom"] = 0.125
        mpl.rcParams["figure.subplot.top"]    = 0.88
        mpl.rcParams["figure.subplot.wspace"] = 0.2
        mpl.rcParams["figure.subplot.hspace"] = 0.2
    """
    #-------------------------
    left   = mpl.rcParams["figure.subplot.left"]   if left   is None else left   # 0.125
    right  = mpl.rcParams["figure.subplot.right"]  if right  is None else right  # 0.9
    bottom = mpl.rcParams["figure.subplot.bottom"] if bottom is None else bottom # 0.125
    top    = mpl.rcParams["figure.subplot.top"]    if top    is None else top    # 0.88
    wspace = mpl.rcParams["figure.subplot.wspace"] if wspace is None else wspace # 0.2
    hspace = mpl.rcParams["figure.subplot.hspace"] if hspace is None else hspace # 0.2
    #-------------------------
    if scale_wspace is not None:
        wspace = scale_wspace*wspace
    if scale_hspace is not None:
        hspace = scale_hspace*hspace
    #-------------------------
    # The idea of a margin only really makes sense if the given edge is between (0,1)
    # Otherwise, there is no margin at all (as the figure extends out past the bounds of the page)
    # If the edge is outside of the page, the scale_margin will have no effect
    if scale_margin_left is not None:
        if left<=0:
            print('scale_margin_left will have no effect as left<=0')
        else:
            left = scale_margin_left*left
    if scale_margin_bottom is not None:
        if bottom<=0:
            print('scale_margin_bottom will have no effect as bottom<=0')
        else:
            bottom = scale_margin_bottom*bottom  
    #-----
    # These are slightly different, as the referece point is 1, not 0 (as above)
    # (i.e., I want to scale the margin size, not the position)
    if scale_margin_right is not None:
        if right>=1:
            print('scale_margin_right will have no effect as right>=1')
        else:
            right = right - scale_margin_right*(1.0-right)
    if scale_margin_top is not None:
        if top>=1:
            print('scale_margin_top will have no effect as top>=1')
        else:
            top = top - scale_margin_top*(1.0-top)
    #-------------------------
    return {'left'   : left, 
            'right'  : right, 
            'bottom' : bottom, 
            'top'    : top, 
            'wspace' : wspace, 
            'hspace' : hspace}
            
def adjust_subplots_args(fig, subplots_adjust_args):
    r"""
    In the past, I typically have called plt.subplots_adjust(**subplots_adjust_args)
    But, it seems calling subplots_adjust on a figure object works as well, and allows
      me to pass in the figure object to this function
    """
    fig.subplots_adjust(**subplots_adjust_args)
    return fig
            
def get_subplot_layout_params(fig):
    r"""
    Returns a dict object containing left, right, bottom, top
    """
    #-------------------------
    # THERE HAS TO BE AN EASIER WAY TO GET THIS!!!!!!!!!
    #-------------------------
    left_coll=[]
    right_coll=[]
    bottom_coll=[]
    top_coll=[]
    for ax_i in fig.axes:
        #----------
        ax_fig_bbox = ax_i.get_position()
        #----------
        left   = ax_fig_bbox.x0
        right  = ax_fig_bbox.x1
        bottom = ax_fig_bbox.y0
        top    = ax_fig_bbox.y1    
        #----------
        left_coll.append(left)
        right_coll.append(right)
        bottom_coll.append(bottom)
        top_coll.append(top)
    #-------------------------
    left = np.min(left_coll)
    right = np.max(right_coll)
    bottom = np.min(bottom_coll)
    top = np.max(top_coll)
    #-------------------------
    return {
        'left':left, 
        'right':right, 
        'bottom':bottom, 
        'top':top
    }            

            
def get_subplots_adjust_args_std_3x1(
    left=None, 
    right=None, 
    bottom=None, 
    top=None, 
    wspace=None, 
    hspace=None, 
    
    scale_margin_left=None, 
    scale_margin_right=None, 
    scale_margin_bottom=None, 
    scale_margin_top=None, 
    scale_wspace=None, 
    scale_hspace=None  
):
    r"""
    STD VALUES 3x1:
        left   = 0.075
        right  = 0.975

        bottom = 0.075
        top    = 0.95

        wspace = mpl.rcParams["figure.subplot.wspace"]
        hspace = 0.5
    """
    #-------------------------
    left   = 0.075                                 if left   is None else left
    right  = 0.975                                 if right  is None else right
    bottom = 0.075                                 if bottom is None else bottom
    top    = 0.95                                  if top    is None else top
    wspace = mpl.rcParams["figure.subplot.wspace"] if wspace is None else wspace
    hspace = 0.5                                   if hspace is None else hspace

    #-------------------------
    subplots_adjust_args = get_subplots_adjust_args(left=left, right=right, 
                                                    bottom=bottom, top=top, 
                                                    wspace=wspace, hspace=hspace, 
                                                    scale_margin_left=scale_margin_left, scale_margin_right=scale_margin_right, 
                                                    scale_margin_bottom=scale_margin_bottom, scale_margin_top=scale_margin_top, 
                                                    scale_wspace=scale_wspace, scale_hspace=scale_hspace)
    return subplots_adjust_args
    
def get_subplots_adjust_args_std_3x2(
    left=None, 
    right=None, 
    bottom=None, 
    top=None, 
    wspace=None, 
    hspace=None, 
    
    scale_margin_left=None, 
    scale_margin_right=None, 
    scale_margin_bottom=None, 
    scale_margin_top=None, 
    scale_wspace=None, 
    scale_hspace=None  
):
    r"""
    !!!!!!!!!! Lazily set exactly same as 3x1, may want to change in future
    
    STD VALUES 3x2:
        left   = 0.075
        right  = 0.975

        bottom = 0.075
        top    = 0.95

        wspace = mpl.rcParams["figure.subplot.wspace"]
        hspace = 0.5
    """
    #-------------------------
    left   = 0.075                                 if left   is None else left
    right  = 0.975                                 if right  is None else right
    bottom = 0.075                                 if bottom is None else bottom
    top    = 0.95                                  if top    is None else top
    wspace = mpl.rcParams["figure.subplot.wspace"] if wspace is None else wspace
    hspace = 0.5                                   if hspace is None else hspace

    #-------------------------
    subplots_adjust_args = get_subplots_adjust_args(left=left, right=right, 
                                                    bottom=bottom, top=top, 
                                                    wspace=wspace, hspace=hspace, 
                                                    scale_margin_left=scale_margin_left, scale_margin_right=scale_margin_right, 
                                                    scale_margin_bottom=scale_margin_bottom, scale_margin_top=scale_margin_top, 
                                                    scale_wspace=scale_wspace, scale_hspace=scale_hspace)
    return subplots_adjust_args
    
def get_subplots_adjust_args_std_2x1(
    left=None, 
    right=None, 
    bottom=None, 
    top=None, 
    wspace=None, 
    hspace=None, 
    
    scale_margin_left=None, 
    scale_margin_right=None, 
    scale_margin_bottom=None, 
    scale_margin_top=None, 
    scale_wspace=None, 
    scale_hspace=None  
):
    #-------------------------
    r"""
    STD VALUES 2x1:
        left   = 0.075
        right  = 0.975

        bottom = 0.075
        top    = 0.95

        wspace = mpl.rcParams["figure.subplot.wspace"]
        hspace = 0.35
    """
    #-------------------------
    left   = 0.075                                 if left   is None else left
    right  = 0.975                                 if right  is None else right
    bottom = 0.075                                 if bottom is None else bottom
    top    = 0.95                                  if top    is None else top
    wspace = mpl.rcParams["figure.subplot.wspace"] if wspace is None else wspace
    hspace = 0.35                                  if hspace is None else hspace
    #------------------
    subplots_adjust_args = get_subplots_adjust_args(left=left, right=right, 
                                                    bottom=bottom, top=top, 
                                                    wspace=wspace, hspace=hspace, 
                                                    scale_margin_left=scale_margin_left, scale_margin_right=scale_margin_right, 
                                                    scale_margin_bottom=scale_margin_bottom, scale_margin_top=scale_margin_top, 
                                                    scale_wspace=scale_wspace, scale_hspace=scale_hspace)
    return subplots_adjust_args
    
    
def rotate_ticklabels(
    ax, 
    rotation, 
    axis='x', 
    ha=None, 
    va=None
):
    r"""
    If labels are numeric (e.g., when binning in x and plotting counts in y), the simpler method
      of using set_x(y)ticklabels doesn't work.
    In this case, instead of throwing an error, a warning is output.
    To make this warning act as an error in the try/except block, I use warnings.filterwarnings('error').
    Then, if 'error' thrown, iterate through all tick labels and set attributes manually
    
    ha: horizontal alignment
        If None, use alignment taken from first label
        
    va: vertical alignment
        If None, use alignment taken from first label
    """
    #-------------------------
    assert(axis in ['x', 'y', 'xy'])
    #-------------------------
    if axis=='xy':
        rotate_ticklabels(
            ax=ax, 
            rotation=rotation, 
            axis='x', 
            ha=ha, 
            va=va
        )
        #-----
        rotate_ticklabels(
            ax=ax, 
            rotation=rotation, 
            axis='y', 
            ha=ha, 
            va=va
        )
        return
    #--------------------------------------------------
    warnings.filterwarnings('error')
    #-------------------------
    if axis=='x':
        axis_i = ax.xaxis
    elif axis=='y':
        axis_i = ax.yaxis
    else:
        assert(0)
    #-------------------------
    if ha is None and len(axis_i.get_ticklabels())>0:
        ha = axis_i.get_ticklabels()[0].get_horizontalalignment()
    if va is None and len(axis_i.get_ticklabels())>0:
        va = axis_i.get_ticklabels()[0].get_verticalalignment()
    #-----
    try:
        axis_i.set_ticklabels(
            axis_i.get_ticklabels(), 
            rotation=rotation, 
            ha=ha, 
            va=va
        );
    except:
        for tick in axis_i.get_ticklabels():
            tick.set_rotation(rotation);
            tick.set_horizontalalignment(ha);
            tick.set_verticalalignment(va);
    #-------------------------
    warnings.resetwarnings()