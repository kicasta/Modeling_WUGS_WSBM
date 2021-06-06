import plot_utils as pltutil
import wsbm as wsbm
import pickle
from collections import Counter
import os
from docopt import docopt


def main():

  # Get the arguments
  args = docopt("""Generates a plot with the graph partition showing a comparison between observed and inferred values
  Usage:
  plot_distributions.py <path_to_graphs> <path_to_pickle> <path_to_save>
      
  Arguments:w

  <path_to_graphs> = path to the directory where all the graphs are stored  
  <path_to_pickle> = path to the pickled dictionary containing the best fit for each graph
  <path_to_save> = path to where the images should be stored   
  """)

  path= args['<path_to_graphs>']
  output_path = args['<path_to_pickle>']
  best_fit = args['<path_to_save>']

  with open(output_path + "g_dist_states", 'rb') as f:
      states = pickle.load(f)

  for g_name,value in states.items():
    if os.path.isfile(best_fit + "detailed_distribution_" + g_name + ".png"):
      print("skipping ", g_name)
      continue
    try:
      g_path = path + g_name

      dist = value[0]
      state = value[1]

      b = wsbm.get_blocks(state)

      graph, g, pos = wsbm.open_graph(g_path)

      edges = g.get_edges([g.ep.orig_weight])

      vertices = g.get_vertices()

      outside_edges = [item for item in edges if 
                      b[vertices[int(item[0])]] != b[vertices[int(item[1])]]]

      outside_weights = [item[2] for item in outside_edges]

      inside_edges = [item for item in edges if 
                      b[vertices[int(item[0])]] == b[vertices[int(item[1])]]]

      inside_weights = [item[2] for item in inside_edges]

      
      inferred_p_inside = wsbm.infer_p(inside_weights, distribution=dist)
      inferred_p_outside = wsbm.infer_p(outside_weights, distribution=dist)

      c = Counter(b.a)
      communities = list(c.values())

      pltutil.plot_values(inside_weights, outside_weights, inferred_p_inside, inferred_p_outside, "joint_" + g_name + ".png", distribution=dist, xticks_shifted=False, path=best_fit)

      partition = wsbm.compute_partition(g, b)

      inferred_ps = dict()

      for k,v in partition.items():
        inferred_ps[k] = wsbm.infer_p(v, distribution=dist)


      blocks = list(b.a)


      for k in inferred_ps.keys():
        in_community = len(k) == 1

        if in_community:
          title = "Block '" + k + "' - Vertex Count: " + str(blocks.count(int(k))) + " - " + "Edge Count: " + str(len(partition[k]))
        else:
          title = "Between Blocks '" + k + "' - Edge Count: " + str(len(partition[k]))
        pltutil.plot_values_oneside(partition[k], inferred_ps[k], in_community, title, fig_title=k, xticks_shifted=False, path=best_fit, distribution=dist)
          
      pltutil.combine_community_plots(list(partition.keys()), len(communities), "detailed_distribution_" + g_name + ".png", path=best_fit)
    except Exception as e:
      print(e)
      continue


if __name__ == 'main':
  main() 