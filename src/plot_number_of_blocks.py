import plot_utils as pltutil
import wsbm as wsbm
import pickle
from docopt import docopt


def main():

  # Get the arguments
  args = docopt("""Generates a plot with the block count for each graph found by the best fit by each of the available distributions, i.e. biomial, poisson and geometric
  Usage:
  plot_number_of_blocks.py <path_to_pickle>
      
  Arguments:w
      
  <path_to_pickle> = path to the pickled dictionary containing the best fit for each graph    
  """)

  output_path = args['<path_to_pickle>']

  with open(output_path + "g_dist_states", 'rb') as f:
      states = pickle.load(f)

  block_counts = {}
  for g,v in states.items():
    state = v[1]
    block_counts[g] = len(set(wsbm.get_blocks(state)))

  blocks_y = [block_counts[k] for k in block_counts.keys()]
  pltutil.plot_single_stat(block_counts.keys(), blocks_y, "Number of Blocks", limy=False)

if __name__ == 'main':
  main() 