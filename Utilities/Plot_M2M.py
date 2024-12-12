#!/usr/bin/env python

"""
Utilities specifically for plotting
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

#-------------------------
def get_color_patches(green_to_blue=True, markersize=7.5, alpha=1.0):
    color_green = 'green'
    if green_to_blue:
        color_green = 'blue'
    patch_g = Line2D([0], [0], marker='o', color=color_green, markerfacecolor=color_green, 
                     markersize=markersize, alpha=alpha, linewidth=0, 
                     label='consistent')
    patch_y =  Line2D([0], [0], marker='o', color='yellow', markerfacecolor='yellow', 
                      markersize=markersize, alpha=alpha, linewidth=0, 
                      label='stat. diff. but within threshold')
    patch_r =  Line2D([0], [0], marker='o', color='red', markerfacecolor='red', 
                      markersize=markersize, alpha=alpha, linewidth=0, 
                      label='sig. different')
    return patch_g, patch_y, patch_r
    
    
def get_m2m_plot_kwargs(**kwargs):
    # Set kwargs if they exist, otherwise set to default values
    plot_kwargs = {'marker':kwargs.get('marker', 'o'), 
                   's':kwargs.get('markersize', 25), 
                   'edgecolors':kwargs.get('edgecolors', 'none'), 
                   'linewidth':kwargs.get('markeredgewidth', kwargs.get('linewidth', 1.0)), 
                   'alpha':kwargs.get('alpha', 1.0), 
                   'zorder':kwargs.get('zorder', 1)}
    return plot_kwargs
    
def get_m2m_line_kwargs(**kwargs):
    # Set kwargs if they exist, otherwise set to default values
    line_kwargs = {'alpha':kwargs.get('alpha', 1.0), 
                   'linewidth':kwargs.get('linewidth', 1.0), 
                   'linestyle':kwargs.get('linestyle', '-'),
                   'color':kwargs.get('linecolor', 'black'),
                   'zorder':kwargs.get('zorder', 1)+1}
    return line_kwargs


def add_plot_m2m(axes, mc_df, 
                 metric_col='metric_name', t_val_col='t_val', color_col='result_color', 
                 green_to_blue=True, plot_line=True, **kwargs):

    mark_within_threshold = kwargs.pop('mark_within_threshold', False)
    # Set kwargs if they exist, otherwise set to default values
    plot_kwargs = get_m2m_plot_kwargs(**kwargs)
    line_kwargs = get_m2m_line_kwargs(**kwargs)
    
    # ---------------------------------------------------------
    
    # Convert color_col to string if need be
    converted_color_col_to_str = False
    if isinstance(mc_df.iloc[0][color_col], Utilities.TTestResultColorType):
        mc_df[color_col] = mc_df[color_col].apply(Utilities.get_color_type_str)
        converted_color_col_to_str = True
        if green_to_blue:
            mc_df.loc[mc_df[color_col]=='green', color_col]='blue'

    if mark_within_threshold:
        # Unfortunately, I cannot simple pass an array as 'marker' to plot
        # Hence this roundabout method
        
        # Get the desired marker_size if set, otherwise set to default=25
        marker_size_og = plot_kwargs.get('s', 25)
        # First, draw the line passing through all data points
        plot_kwargs['s'] = 0
        mc_df.plot(ax=axes, x=metric_col, y=t_val_col, 
                   kind='scatter', color=mc_df[color_col], 
                   **plot_kwargs)        
        plot_kwargs['s'] = marker_size_og
        # Next, draw the data within threshold
        plot_kwargs['marker'] = 'o'
        within_df = mc_df[mc_df['is_within_threshold']==True]
        within_df.plot(ax=axes, x=metric_col, y=t_val_col, 
                       kind='scatter', color=within_df[color_col], 
                       **plot_kwargs)
        # Finally, draw the data not within threshold
        plot_kwargs['marker'] = 'x'
        not_within_df = mc_df[mc_df['is_within_threshold']==False]
        not_within_df.plot(ax=axes, x=metric_col, y=t_val_col, 
                           kind='scatter', color=not_within_df[color_col], 
                           **plot_kwargs)

    else:
        mc_df.plot(ax=axes, x=metric_col, y=t_val_col, 
                   kind='scatter', color=mc_df[color_col], 
                   **plot_kwargs)
    if plot_line:
        mc_df.plot.line(ax=axes, x=metric_col, y=t_val_col, legend=None, **line_kwargs)
    
    # Convert color_col back to TTestResultColorType if converted earlier
    if converted_color_col_to_str:
        if green_to_blue:
            mc_df.loc[mc_df[color_col]=='blue', color_col]='green'
        mc_df[color_col] = mc_df[color_col].apply(Utilities.str_to_color_type)
        
        
def plot_m2m(fig_num, mc_df, 
             metric_col='metric_name', t_val_col='t_val', color_col='result_color', 
             green_to_blue=True, plot_line=True, **kwargs):
    fig, ax = plt.subplots(1, 1, num=fig_num, figsize=[11, 8.5])
    mark_within_threshold = kwargs.pop('mark_within_threshold', False)
    add_plot_m2m(ax, mc_df, 
                 metric_col, t_val_col, color_col, 
                 green_to_blue=green_to_blue, mark_within_threshold=mark_within_threshold, plot_line=plot_line, 
                 **kwargs)
    
    ax.set_xlim(-1, len(mc_df[t_val_col]))  
    ax.set_ylabel('t-test statistic')
    ax.set_xlabel('')
    
    ax.axhline(y=0, color='black', linestyle='-', zorder=0)
    ax.grid(True)
    ax.tick_params(axis='y', direction='in');
    ax.tick_params(axis='x', labelrotation=90, labelsize=7.0, direction='in');
 
    #----- 
    patch_g, patch_y, patch_r = get_color_patches()
    leg_1 = ax.legend(title='t-test statistic', handles=[patch_g, patch_y, patch_r], bbox_to_anchor=(1, 0.975), loc='upper left')
    #-----
    if mark_within_threshold:
        plot_kwargs = get_m2m_plot_kwargs(**kwargs)

        patch_within =  Line2D([0], [0], 
                               label='Within Threshold', marker='o', 
                               color='black', markerfacecolor='black', markeredgecolor='black', 
                               markersize=7.5, alpha=plot_kwargs['alpha'], linewidth=0)

        patch_not_within =  Line2D([0], [0], 
                                   label='Outside Threshold', marker='x', 
                                   color='black', markerfacecolor='black', markeredgecolor='black', 
                                   markersize=7.5, alpha=plot_kwargs['alpha'], linewidth=0)
        #-----
        # Add second legend. leg_1 will be removed from the figure
        leg_2 = ax.legend(title='', handles=[patch_within, patch_not_within], bbox_to_anchor=(1, 0.375), loc='upper left')
        # Manually add the first legend back
        ax.add_artist(leg_1)
    #-----

    plt.subplots_adjust(top=0.925, bottom=0.70, left=0.050, right=0.75)
    fig.suptitle('Metric-to-Metric', fontsize=25, fontweight='bold');
    return fig, ax
    
#---------------------------------------------------------------------
# A vs B    
def split_kwargs_avb(**kwargs):
    # Anything appended explicitly with _a will only be for a
    # Anything appended explicitly with _b will only be for b
    # Anything with no appendix will be for both a and b
    kwargs_a = {}
    kwargs_b = {}
    for key, value in kwargs.items():
        if key[-2:]=='_a':
            kwargs_a[key[:-2]] = value
        elif key[-2:]=='_b':
            kwargs_b[key[:-2]] = value
        else:
            kwargs_a[key] = value
            kwargs_b[key] = value
    return kwargs_a, kwargs_b
    

def add_plot_m2m_avb(ax, 
                     mc_df_a, label_a, 
                     mc_df_b, label_b, 
                     metric_col='metric_name', t_val_col='t_val', color_col='result_color', 
                     green_to_blue=True, plot_line=True, include_legends=True, 
                     **kwargs):
    #--------------------------------
    mark_within_threshold = kwargs.pop('mark_within_threshold', False)
    mark_within_threshold_a = kwargs.pop('mark_within_threshold_a', mark_within_threshold)
    mark_within_threshold_b = kwargs.pop('mark_within_threshold_b', mark_within_threshold)
    #--------------------------------
    exclude_avb_legend = kwargs.pop('exclude_avb_legend', False)
    #--------------------------------
    # Set default color scheme if not already set
    kwargs['marker_a'] = kwargs.get('marker_a', kwargs.get('marker', 'o'))
    kwargs['linecolor_a'] = kwargs.get('linecolor_a', kwargs.get('linecolor', 'darkviolet'))
    kwargs['edgecolors_a'] = kwargs.get('edgecolors_a', kwargs.get('edgecolors', 'darkviolet'))
    
    kwargs['marker_b'] = kwargs.get('marker_b', kwargs.get('marker', 'd'))
    kwargs['linecolor_b'] = kwargs.get('linecolor_b', kwargs.get('linecolor', 'lawngreen'))
    kwargs['edgecolors_b'] = kwargs.get('edgecolors_b', kwargs.get('edgecolors', 'lawngreen'))    
    
    kwargs_a, kwargs_b = split_kwargs_avb(**kwargs)
    #--------------------------------
    add_plot_m2m(ax, mc_df_a, 
                 metric_col, t_val_col, color_col, 
                 green_to_blue=green_to_blue, mark_within_threshold=mark_within_threshold_a, plot_line=plot_line, 
                 **kwargs_a)
    add_plot_m2m(ax, mc_df_b, 
                 metric_col, t_val_col, color_col, 
                 green_to_blue=green_to_blue, mark_within_threshold=mark_within_threshold_b, plot_line=plot_line, 
                 **kwargs_b)
    #--------------------------------
    y_min = min(mc_df_a[t_val_col].min(), mc_df_b[t_val_col].min())
    y_max = max(mc_df_a[t_val_col].max(), mc_df_b[t_val_col].max())
    
    y_min = int(round(y_min/5.))*5 - 5
    y_max = int(round(y_max/5.))*5 + 5
    
    ax.set_ylim([y_min, y_max]);
    ax.set_xlim(-1, len(mc_df_a[t_val_col]))    
    #--------------------------------
    ax.set_ylabel('t-test statistic')
    ax.set_xlabel('')
    
    ax.axhline(y=0, color='black', linestyle='-', zorder=0)
    ax.grid(True)
    ax.tick_params(axis='y', direction='in');
    ax.tick_params(axis='x', labelrotation=90, labelsize=7.0, direction='in');
    #--------------------------------
    patch_alpha = 1.0
    if kwargs_a.get('alpha', 1.0) == kwargs_b.get('alpha', 1.0):
        patch_alpha = kwargs_a.get('alpha', 1.0)
    patch_g, patch_y, patch_r = get_color_patches(alpha = patch_alpha)
    #-----
    plot_kwargs_a = get_m2m_plot_kwargs(**kwargs_a)
    line_kwargs_a = get_m2m_line_kwargs(**kwargs_a)
    patch_a =  Line2D([0], [0], 
                      label=label_a, marker=plot_kwargs_a['marker'], 
                      color=plot_kwargs_a['edgecolors'], markerfacecolor='black', markeredgecolor=plot_kwargs_a['edgecolors'], 
                      markersize=7.5, alpha=plot_kwargs_a['alpha'], linewidth=line_kwargs_a['linewidth'])

    plot_kwargs_b = get_m2m_plot_kwargs(**kwargs_b)
    line_kwargs_b = get_m2m_line_kwargs(**kwargs_b)
    patch_b =  Line2D([0], [0], 
                      label=label_b, marker=plot_kwargs_b['marker'], 
                      color=plot_kwargs_b['edgecolors'], markerfacecolor='black', markeredgecolor=plot_kwargs_b['edgecolors'], 
                      markersize=7.5, alpha=plot_kwargs_b['alpha'], linewidth=line_kwargs_b['linewidth']) 
    #-----    
    if include_legends:
        # Add first legend
        leg_1_bbox_to_anchor_y = kwargs.get('leg_1_bbox_to_anchor_y', 0.975)
        leg_1_size = kwargs.get('leg_1_size', None)
        if leg_1_size is None:
            prop_1 = {}
        else:
            prop_1 = {'size':leg_1_size}
        leg_1 = ax.legend(title='t-test statistic', handles=[patch_g, patch_y, patch_r], 
                          bbox_to_anchor=(1, leg_1_bbox_to_anchor_y), loc='upper left', prop=prop_1)
        plt.setp(leg_1.get_title(), multialignment='center')
        # Add second legend. leg1 will be removed from the figure
        if not exclude_avb_legend:
            leg_2_bbox_to_anchor_y = kwargs.get('leg_2_bbox_to_anchor_y', 0.375)
            leg_2_size = kwargs.get('leg_2_size', None)
            if leg_2_size is None:
                prop_2 = {}
            else:
                prop_2 = {'size':leg_2_size}
            leg_2 = ax.legend(title='', handles=[patch_a, patch_b], 
                              bbox_to_anchor=(1, leg_2_bbox_to_anchor_y), loc='upper left', prop=prop_2)
            # Manually add the first legend back
            ax.add_artist(leg_1)
    
    if mark_within_threshold:
        if include_legends:
            plot_kwargs = get_m2m_plot_kwargs(**kwargs)

            patch_within =  Line2D([0], [0], 
                                   label='Within Threshold', marker='o', 
                                   color='black', markerfacecolor='black', markeredgecolor='black', 
                                   markersize=7.5, alpha=plot_kwargs['alpha'], linewidth=0)

            patch_not_within =  Line2D([0], [0], 
                                       label='Outside Threshold', marker='x', 
                                       color='black', markerfacecolor='black', markeredgecolor='black', 
                                       markersize=7.5, alpha=plot_kwargs['alpha'], linewidth=0)

            if not exclude_avb_legend:
                leg_1 = ax.legend(title='t-test statistic', handles=[patch_g, patch_y, patch_r], bbox_to_anchor=(1, 1), loc='upper left')
                leg_2 = ax.legend(title='', handles=[patch_within, patch_not_within], bbox_to_anchor=(1, 0.475), loc='upper left')
                leg_3 = ax.legend(title='', handles=[patch_a, patch_b], bbox_to_anchor=(1, 0.175), loc='upper left')
                ax.add_artist(leg_2)
            else:
                leg_1 = ax.legend(title='t-test statistic', handles=[patch_g, patch_y, patch_r], bbox_to_anchor=(1, 1), loc='upper left')
                leg_2 = ax.legend(title='', handles=[patch_within, patch_not_within], bbox_to_anchor=(1, 0.475), loc='upper left')
                ax.add_artist(leg_1)
  
def plot_m2m_avb(fig_num, 
                 mc_df_a, label_a, 
                 mc_df_b, label_b, 
                 metric_col='metric_name', t_val_col='t_val', color_col='result_color', 
                 green_to_blue=True, plot_line=True, 
                 **kwargs):
    fig, ax = plt.subplots(1, 1, num=fig_num, figsize=[11, 8.5])
    #--------------------------------
    add_plot_m2m_avb(ax, 
                     mc_df_a, label_a, 
                     mc_df_b, label_b, 
                     metric_col, t_val_col, color_col, 
                     green_to_blue, plot_line, 
                     **kwargs)
    #-----
    left = 0.050
    right = 0.75
    plt.subplots_adjust(top=0.925, bottom=0.70, left=left, right=right)
    sup_center = left +(right-left)/2.0
    fig.suptitle('Metric-to-Metric', x=sup_center, ha='center', fontsize=25, fontweight='bold');
    return fig, ax