#!/usr/bin/env python

"""
Utilities specifically for plotting boxplots with seaborn
  Using Matplotlib without seaborn allows for me flexibility,
  so if this doesn't do what you need, see Plot_Box.py
"""

#---------------------------------------------------------------------
import os
import sys

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import stats
#---------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
#---------------------------------------------------------------------
import Utilities
import Utilities_xml
import Utilities_df
#---------------------------------------------------------------------

#******************************** Handling of kwargs ****************************************
def get_kwargs_box_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # NOTE: Color of individual boxes set by palette argument of sns.boxplot
    # Acceptable kwargs are:
    #   'alpha'
    #   'linestyle'
    #   'edgecolor' (set via 'edgecolor' or 'color')
    #   'hatch'
    #   'fill'
    return_kwargs = {}
    return_kwargs['alpha'] = kwargs.get('alpha', 1.0)
    return_kwargs['linestyle'] = kwargs.get('linestyle', '-')
    if kwargs.get('edgecolor', None) is not None:
        return_kwargs['edgecolor'] = kwargs['edgecolor']
    else:
        return_kwargs['edgecolor'] = kwargs.get('color', 'black')
    return_kwargs['hatch'] = kwargs.get('hatch', None)
    return_kwargs['fill'] = True if return_kwargs['hatch'] is None else False
    
    return return_kwargs

    
def get_kwargs_whisker_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'color'
    #   'linestyle'
    return_kwargs = {}
    return_kwargs['color'] = kwargs.get('color', 'black')
    return_kwargs['linestyle'] = kwargs.get('linestyle', '-')
    
    return return_kwargs

def get_kwargs_mean_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'color'
    #   'marker'
    #   'markerfacecolor'
    #   'markeredgecolor'
    return_kwargs = {}
    return_kwargs['color'] = kwargs.get('color', 'black')
    return_kwargs['marker'] = kwargs.get('marker', 'X')
    return_kwargs['markerfacecolor'] = kwargs.get('markerfacecolor', return_kwargs['color'])
    return_kwargs['markeredgecolor'] = kwargs.get('markeredgecolor', 'white')

    return return_kwargs

    
def get_kwargs_median_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'color'
    return_kwargs = {}
    return_kwargs['color'] = kwargs.get('color', 'black')

    return return_kwargs

    
def get_kwargs_flier_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'marker'
    #   'markersize'
    #   'markerfacecolor'
    #   'markeredgecolor'
    return_kwargs = {}
    return_kwargs['marker'] = kwargs.get('marker', 'o')
    return_kwargs['markersize'] = kwargs.get('markersize', 5)
    if kwargs.get('markerfacecolor', None) is not None:
        return_kwargs['markerfacecolor'] = kwargs['markerfacecolor']
    else:
        return_kwargs['markerfacecolor'] = kwargs.get('color', 'black')
    return_kwargs['markeredgecolor'] = kwargs.get('markeredgecolor', 'white')

    return return_kwargs


def get_kwargs_cap_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'color'
    return_kwargs = {}
    return_kwargs['color'] = kwargs.get('color', 'black')

    return return_kwargs


def get_kwargs_strip_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'palette'
    #   'size'
    #   'alpha'
    #   'edgecolor'
    #   'linewidth'
    return_kwargs = {}
    return_kwargs['palette'] = kwargs.get('palette', 'Spectral')
    return_kwargs['size'] = kwargs.get('size', 3)
    return_kwargs['alpha'] = kwargs.get('alpha', 0.5)
    return_kwargs['edgecolor'] = kwargs.get('edgecolor', 'black')
    return_kwargs['linewidth'] = kwargs.get('linewidth', 1)
    return_kwargs['hue'] = kwargs.get('hue', None)
    return_kwargs['jitter'] = kwargs.get('jitter', True) #jitter can be float also to quantify amount of jitter
    return_kwargs['dodge'] = kwargs.get('dodge', False if return_kwargs['hue'] is None else True)

    return return_kwargs
    
    
def get_kwargs_patch_sns(**kwargs):
    # For now, make clean kwargs, only housing expected kwargs
    # Call AFTER split_kwargs_box_and_strip_sns
    # Acceptable kwargs are:
    #   'label'
    #   'alpha'
    #   'linestyle'
    #   'color'
    #   'fill'
    #   'hatch'
    return_kwargs = {}
    return_kwargs['label'] = kwargs.get('label', '')
    return_kwargs['alpha'] = kwargs.get('alpha', 1.0)
    return_kwargs['linestyle'] = kwargs.get('linestyle', '-')
    return_kwargs['color'] = kwargs.get('color', 'black')    
    return_kwargs['hatch'] = kwargs.get('hatch', None)
    return_kwargs['fill'] = True if return_kwargs['hatch'] is None else False

    return return_kwargs
    
#-----*****-----*****-----*****-----*****-----*****-----*****
def split_kwargs_box_and_strip_sns(**kwargs):
    # Anything appended explicitly with _box will only be for boxprops
    # Anything appended explicitly with _whisker will only be for whiskerprops
    # Anything appended explicitly with _mean will only be for meanprops
    # Anything appended explicitly with _median will only be for medianprops
    # Anything appended explicitly with _flier will only be for flierprops
    # Anything appended explicitly with _cap will only be for capprops
    # Anything appended explicitly with _strip will only be for stripplot
    # Anything appended explicitly with _patch will only be for mpatches.Patch
    # Anything with no appendix will be for all
    kwargs_box     = {}
    kwargs_whisker = {}
    kwargs_mean    = {}
    kwargs_median  = {}
    kwargs_flier   = {}
    kwargs_cap     = {}
    kwargs_strip   = {}
    kwargs_patch   = {}
    for key, value in kwargs.items():
        if key.endswith('_box'):
            kwargs_box[key[:-len('_box')]] = value
        elif key.endswith('_whisker'):
            kwargs_whisker[key[:-len('_whisker')]] = value
        elif key.endswith('_mean'):
            kwargs_mean[key[:-len('_mean')]] = value        
        elif key.endswith('_median'):
            kwargs_median[key[:-len('_median')]] = value
        elif key.endswith('_flier'):
            kwargs_flier[key[:-len('_flier')]] = value            
        elif key.endswith('_cap'):
            kwargs_cap[key[:-len('_cap')]] = value
        elif key.endswith('_strip'):
            kwargs_strip[key[:-len('_strip')]] = value
        elif key.endswith('_patch'):
            kwargs_patch[key[:-len('_patch')]] = value  
        else:           
            kwargs_box[key]     = value
            kwargs_whisker[key] = value
            kwargs_mean[key]    = value
            kwargs_median[key]  = value
            kwargs_flier[key]   = value
            kwargs_cap[key]     = value
            kwargs_strip[key]   = value
            kwargs_patch[key]   = value

    return {'kwargs_box':     get_kwargs_box_sns(**kwargs_box),
            'kwargs_whisker': get_kwargs_whisker_sns(**kwargs_whisker),
            'kwargs_mean':    get_kwargs_mean_sns(**kwargs_mean),
            'kwargs_median':  get_kwargs_median_sns(**kwargs_median),
            'kwargs_flier':   get_kwargs_flier_sns(**kwargs_flier),
            'kwargs_cap':     get_kwargs_cap_sns(**kwargs_cap), 
            'kwargs_strip':   get_kwargs_strip_sns(**kwargs_strip), 
            'kwargs_patch':   get_kwargs_patch_sns(**kwargs_patch)}

#******************************** Plotting ****************************************
def setup_axes(ax, **kwargs):
    include_labels = kwargs.get('include_labels', 'passive') #include, exclude, or passive
    xtick_labelsize = kwargs.get('xtick_labelsize', 8.0)
    xtick_labelrotation = kwargs.get('xtick_labelrotation', 90)
    xtick_direction = kwargs.get('xtick_direction', 'in')
    xtick_labels = kwargs.get('xtick_labels', ax.get_xticklabels())
    xlabel = kwargs.get('xlabel', ax.get_xlabel())
    
    ylabel = kwargs.get('ylabel', ax.get_ylabel())
    
    grid = kwargs.get('grid', False)
    
    if include_labels=='include':
        ax.tick_params(axis='x', 
                       labelrotation=xtick_labelrotation, 
                       labelsize=xtick_labelsize, 
                       direction=xtick_direction);
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    elif include_labels=='exclude':
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.set(xlabel=None)
        ax.set(ylabel=None)
    else:
        assert(include_labels=='passive')
        
    ax.grid(grid)
    
def add_means_to_axes(ax, x_group_col, y_metric_col, df, avg_type='normal', **kwargs):
    assert(avg_type=='normal' or avg_type=='grubbs' or avg_type=='trim')
    kwargs_final = {}
    kwargs_final['marker']     = kwargs.get('marker', '+' if avg_type=='trim' else '*' if avg_type=='grubbs' else 'X')
    kwargs_final['color']      = kwargs.get('color', 'cyan' if avg_type=='trim' else 'magenta' if avg_type=='grubbs' else 'black')
    kwargs_final['markeredgecolor'] = kwargs.get('markeredgecolor', kwargs_final['color'])
    kwargs_final['alpha']      = kwargs.get('alpha', 1.0)
    kwargs_final['markersize'] = kwargs.get('markersize', 5)
    kwargs_final['linewidth']  = kwargs.get('linewidth', 0)
    kwargs_final['zorder']     = kwargs.get('zorder', 100)
    #-------------------------
    avgs_df = Utilities_df.build_avgs_df(df, x_group_col, [y_metric_col], avg_type)
    #-------------------------
    x_vals = avgs_df[x_group_col].tolist()
    y_vals = avgs_df[y_metric_col].tolist()
    ax.plot(x_vals, y_vals, **kwargs_final)
    
    
def add_df_box_plots_to_axes_sns(ax, x_group_col, y_metric_col, df, 
                                 **kwargs):
    # x_group_col is the df column used to group the data into differnt x-value bins
    # y_metric_col is the metric we are interested in visualizing
    # hue can be used to further split each x_group
    
    # First, general kwargs
    hue = kwargs.get('hue', None)
    showfliers = kwargs.get('showfliers', True)
    showmeans  = kwargs.get('showmeans', True)
    palette    = kwargs.get('palette', 'Spectral')
    zorder     = kwargs.get('zorder', 1)
    
    # Split the kwargs
    # For each key in dict_of_kwargs, this also sets kwargs if they exist, 
    # otherwise set to default values
    dict_of_kwargs = split_kwargs_box_and_strip_sns(**kwargs)
    
    box_plot = sns.boxplot(ax=ax, x=x_group_col, y=y_metric_col, hue=hue, data=df, 
                           showfliers=showfliers, 
                           showmeans=showmeans, 
                           palette=palette, 
                           zorder=zorder, 
                           boxprops=dict_of_kwargs['kwargs_box'], 
                           whiskerprops=dict_of_kwargs['kwargs_whisker'],
                           meanprops=dict_of_kwargs['kwargs_mean'],
                           medianprops=dict_of_kwargs['kwargs_median'],
                           flierprops = dict_of_kwargs['kwargs_flier'], 
                           capprops=dict_of_kwargs['kwargs_cap'])
    setup_axes(ax, **kwargs)
    
    patch = mpatches.Patch(**dict_of_kwargs['kwargs_patch'])
    return box_plot, patch


def add_df_strip_plots_to_axes_sns(ax, x_group_col, y_metric_col, df, 
                                   **kwargs):
    # x_group_col is the df column used to group the data into differnt x-value bins
    # y_metric_col is the metric we are interested in visualizing
    # hue can be used to further split each x_group
    
    # Split the kwargs
    # For each key in dict_of_kwargs, this also sets kwargs if they exist, 
    # otherwise set to default values
    dict_of_kwargs = split_kwargs_box_and_strip_sns(**kwargs)
    
    strip_plot = sns.stripplot(ax=ax, x=x_group_col, y=y_metric_col, data=df, 
                               **dict_of_kwargs['kwargs_strip'])
    setup_axes(ax, **kwargs)
    
    patch = mpatches.Patch(**dict_of_kwargs['kwargs_patch'])
    return strip_plot, patch


def add_df_box_and_strip_plots_to_axes_sns(ax, x_group_col, y_metric_col, df, 
                                           **kwargs):
    # x_group_col is the df column used to group the data into differnt x-value bins
    # y_metric_col is the metric we are interested in visualizing
    # hue can be used to further split each x_group
    
    # passive should be used when adding plots on top of already drawn plots
    kwargs['include_labels'] = kwargs.get('include_labels', 'passive') #include, exclude, or passive
    kwargs['showfliers'] = False
    box_plot, patch = add_df_box_plots_to_axes_sns(ax=ax, 
                                                   x_group_col=x_group_col, 
                                                   y_metric_col=y_metric_col, 
                                                   df=df, 
                                                   **kwargs)
    kwargs['include_labels'] = 'passive'
    add_df_strip_plots_to_axes_sns(ax=ax, 
                                   x_group_col=x_group_col, 
                                   y_metric_col=y_metric_col, 
                                   df=df, 
                                   **kwargs)   
    return box_plot, patch


def build_df_box_plot_sns(fig_num, x_group_col, y_metric_col, df, 
                          **kwargs):
    fig, ax = plt.subplots(1, 1, num=fig_num, figsize=[11, 8.5])
    add_df_box_plots_to_axes_sns(ax, x_group_col, y_metric_col, df, **kwargs)
    return fig, ax

def build_df_box_and_strip_plots_sns(fig_num, x_group_col, y_metric_col, df, 
                                     **kwargs):
    fig, ax = plt.subplots(1, 1, num=fig_num, figsize=[11, 8.5])
    kwargs['include_labels'] = 'include'
    add_df_box_and_strip_plots_to_axes_sns(ax, x_group_col, y_metric_col, df, 
                                           **kwargs)
    return fig, ax
