# WSBM_WUG_Modelling
Source Code and Experiments for the paper: Modeling Sense Structure in Word Usage Graphswith the Weighted Stochastic Block Model

## Description

In this project we present 2 function files:

* **wsbm.py** contains util functions to handle wsbms, for instance, create a graph-tool graph out of a wug *sem_eval_2_graph_tool*, fit a graph-tool graph given a distribution *minimize* or find the best distribution for a given graph *find_best_distribution* by selecting the fitted model with distribution that yields the smallest entropy.

* **plot_utils.py** contains util functions to handle all relative to plots, for instance plots of graph-tool graphs *draw*, plots of histograms showing the relation of a graph distribution between and inside blocks *plot_values_oneside* and *plot_values*, as well as a function to plot the histogram showing amount of distributions that best fit the graphs *plot_dist_dist* and a function to combine the plots of *plot_values_oneside* into a detailed image showing all the distribution combinations *combine_community_plots*.  

<br>
We also present a the scripts used to fit the WUGs and generate the different plots:

* **save_semeval_wsbm_info.py** computes the best distribution describing the graphs and pickles two dictionaries: one containing a pro graph the name of this distribution and the graph partition based on this distribution *graph-tool state*, and another containing the accuracy of this partition.

* **draw_wugs.py** saves the fitted wug images

* **plot_accuracies.py** and **plot_number_of_blocks.py** generate the plots for the accuracy and the number of blocks of the best fit (using the values saved in *save_semeval_wsbm_info.py*) 

* **plot_distributions.py** generates the plots with the distribution relations between and inside blocks as well as the prediction that the model would make based on the best fit distribution parameters. WARNING this script takes a long time to run, since it infer the parameters of all posible combinations (inside blocks and between blocks) for all graphs

* **plot_stats_best.py** generates the plot with the amount of distributions best fitting the graphs

## Usage

Note that the script parameters vary, but are always paths to save the output of the script or to get the saved data as input.

### Example of use:

`python draw_wugs.py <path_to_graphs> <path_to_pickled_data> <output_path>`
