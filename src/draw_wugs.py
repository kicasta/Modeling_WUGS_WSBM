import plot_utils as pltutil
import wsbm as wsbm
import pickle
from docopt import docopt


def main():

  # Get the arguments
  args = docopt("""Generates a plot with the graph partition showing a comparison between observed and inferred values
  Usage:
  draw_wugs.py <path_to_graphs> <path_to_pickle> <path_to_save>
      
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
    try:
      g_path = path + g_name

      dist = value[0]
      state = value[1]

      b = wsbm.get_blocks(state)

      graph, g, pos = wsbm.open_graph(g_path)

      pltutil.draw(g, best_fit + g_name + ".png", g.own_property(b), show_img=False)
    except Exception as e:
      print(e)
      continue

if __name__ == 'main':
  main() 
