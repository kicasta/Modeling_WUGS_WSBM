# WSBM_WUG_Modelling
Source Code and Experiments for this paper:

- Dominik Schlechtweg, Enrique Castaneda, Jonas Kuhn, and Sabine Schulte im Walde. 2021. **Modeling Sense Structure in Word Usage Graphs with the Weighted Stochastic Block Model**. In Proceedings of the 10th Joint Conference on Lexical and Computational Semantics.

If you use this repository for academic research, please [cite](#bibtex) us.


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

For a better understanding of the code, how it is structured and so on, we put in some way all the scripts in a notebook in the example directory. Notice that not always the code in the script is 100% the same, since we wanted to keep things simple in the notebook. It is also configured to run on colab in case you want to just test something. 

```
@inproceedings{Schlechtweg2021wsbm,
	Author = {Schlechtweg, Dominik and Castaneda, Enrique and Kuhn, Jonas and {Schulte im Walde}, Sabine},
	Booktitle = {{Proceedings of the 10th Joint Conference on Lexical and Computational Semantics}},
	Title = {Modeling Sense Structure in Word Usage Graphs with the Weighted Stochastic Block Model},
	Year = {2021}
}
```

