import plot_utils as pltutil
import pickle
from docopt import docopt


def main():

    # Get the arguments
    args = docopt("""Generates a plot with the amount of graphs best fit by each of the available distributions, i.e. biomial, poisson and geometric
    Usage:
    plot_stats_best.py <path_to_pickle>
        
    Arguments:w
        
    <path_to_pickle> = path to the pickled dictionary containing the best fit for each graph   
    """)

    output_path = args['<path_to_pickle>']

    with open(output_path + "g_dist_states", 'rb') as f:
        states = pickle.load(f)

    dists = [d for d,s in states.values()]
    dist_count = {d:dists.count(d) for d in set(dists)}

    dist_labels = dist_count.keys()
    dist_values = [dist_count[d] for d in dist_labels]
    dist_labels = [d.split("-")[1] for d in dist_labels]

    pltutil.plot_dist_dist(dist_labels, dist_values)

if __name__ == 'main':
  main() 