#plots.py

# os library or module allows you to access 
# commandline functions from python. Such as making a new directory, 
# changing or navigating between directories, copy files, change file names, etc
import os
import pandas as pd
# numpy is a math data library
import numpy as np

# matplotlib is used for plotting and graphs
import matplotlib.pyplot as plt

def plot_ts_scatter(df, s = 75, figsize = (40, 20), save_fig = False, pp = None):
# s=75 is size of the points in the scatter
# notice that figsice is defined in the function so this is the default sice of any figure -
# we create by calling this function
# save_fit false means we won't save the fig in a pp (PDF)    
        
# Create plot for every unique pair of variables
# plot_vars is the plot variables or keys of the plot
    plot_vars = list(df.keys())
# cycle through each variable for x value
    for x in plot_vars:
# cycle again for y-value
        for y in plot_vars:
# check to make sure x variable does not equal y variable            
            if x != y:
# notice we already defined a default figsice in the plots function
# so we'll use that default figsice in the subplot                
                fig, ax = plt.subplots(figsize = figsize)
# Create list of years from index Year will be represented by color
# we'll create the year value in the df
# years will be the c value
                if "Year" not in df.keys():
# create list from index, convert each index value to string
# only include first 4 characters which is the year  
# create an integer from those characters                 
                    df["Year"] = [int(str(ind)[:4]) for ind in df.index] 
# assign x & y values, s stands for size of points which we already defined -
# in the function parameter. c represents color so we want to use the year column
# to color the data                 
                df.plot.scatter(x = x, y = y, s = s, ax = ax, 
                                c = "Year", cmap = "viridis") #cmap deals with colormaps in matplotlib
                
# Turn the text on the x-axis so that it reads vertically
                ax.tick_params(axis='x', rotation=90)
                
# Get rid of tick lines perpendicular to both axis for aesthetic
# by setting the length of the tickmarks to cero
                ax.tick_params('both', length=0, which='both')
                
# save image if PdfPages object was passed then try to create a new folder
# if the folder doesn't exist, create it. If it exists, save the fig
                if save_fig:
                    try:                       
                        os.mkdir("plots")
                    except:
                        pass
# identify directory to save figure and save the plots folder then name the file. 
# The first 12 chars of the first var, first 12 chars of the second var, identify the color
# save as .png as well as pdf
                    directory = "plots/" + x[:12] + " " + y[:12] + "c=Year"                  
                    plt.savefig(directory.replace(":", "-") + ".png")
                if pp != None: pp.savefig(fig, bbox_inches = "tight")
       
              
  
      

    # function for line plots        
                
def plot_lines(df, linewidth = 1, figsize = (40,20), 
               legend = True, pp = None):
    fig, ax = plt.subplots(figsize = figsize)
# If no secondary y-axis, plot all variables at once
    df.plot.line(linewidth = linewidth, ax = ax, legend = legend)
# turn the text on the x-axis so taht it reads vertically
    ax.tick_params(axis ="x", rotation = 90)
# get rid of tick lines
    ax.tick_params("both", length=0, which = "both")
# transform y-axis values from scientific notations to integers
    vals = ax.get_yticks()
    vals = [int(x) for x in vals]
    ax.set_yticklabels(vals)
    
# format image filename by replacing unwanted characters
    remove_chars = "[]:$'\\"
    filename = str(list(df.keys()))
    for char in remove_chars:
        filename=filename.replace(char, "")
#Save file and also avoid cutting off text
    plt.savefig(filename[:50] + "line.png", 
                bbox_inches = "tight") #avoids cutting off text
    if pp != None: pp.savefig(fig, box_inches = "tight")
    
    
    
    
    
      # function for stacked lines
def plot_stacked_lines(df, plot_vars, linewidth = 1, figsize = (40, 20),
                       pp = None, total_var = False, title = False):
    fig, ax = plt.subplots(figsize = figsize)
    
# df.plot.area() creates a stacked plot    
    df[plot_vars].plot.area(stacked = True, linewidth = linewidth, ax = ax)
    
# change y als from mil to tril   
    if total_var != False:
        df[total_var].plot.line(linewidth = linewidth, ax = ax, 
                                c = "k", label = total_var, ls="--")
        
 # loc = 2 is the top left column 
 # ncol=2 meeas the legend will have 2 columns of names      
    ax.legend(loc =2, ncol = 2)
    if title != False:
        plt.title(title)