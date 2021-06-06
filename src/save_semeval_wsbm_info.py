import os
import wsbm as wsbm
import pickle
from docopt import docopt


def main():

  # Get the arguments
  args = docopt("""Fits all graphs in a directory finding the distribution that best describe each one (See the function 'find_best_distribution' in wsbm.py). 
  Saves the distribution name as well as the corresponding partition in a pickled dictionary.
  Usage:
  save_semeval_wsbm_info.py <path_to_graphs> <path_to_pickle>
      
  Arguments:w
      
  <path_to_graphs> = path to the directory where all the graphs are stored
  <path_to_pickle> = output path, to save the pickled dictionaries    
  """)

  in_path= args['<path_to_graphs>']
  output_path = args['<path_to_pickle>']

  graphs = os.listdir(in_path)

  #{g:(dist,state)} - contains pro graph its best distribution and its partition according to said distribution
  states = {}
  #{g:acc} - contains the accuracy for the best fit
  accuracies = {}

  for g in graphs:
    if g.startswith("."):
      continue
    g_path = in_path + g
    
    graph, s_gt, pos = wsbm.open_graph(g_path)

    dist, state = wsbm.find_best_distribution(s_gt)
    b = wsbm.get_blocks(state)

    states[g] = (dist, state)

    mri, purity, acc = wsbm.partition_and_stat_gt(graph, s_gt, b, verbose=False)

    accuracies[g] = acc

  #Pickle both dictionaries
  pickle.dump(states, open(output_path + "g_dist_states", "wb"))
  pickle.dump(accuracies, open(output_path + "g_accuracies", "wb"))

if __name__ == 'main':
  main()