import plot_utils as pltutil
import pickle
from docopt import docopt


def main():

    # Get the arguments
    args = docopt("""Generates a plot with the accuracy for each graph found by the best fit by each of the available distributions, i.e. biomial, poisson and geometric
    Usage:
    plot_accuracies.py <path_to_pickle>
        
    Arguments:w
        
    <path_to_pickle> = path to the pickled dictionary containing the accuracy of the best fit for each graph    
    """)

    output_path = args['<path_to_pickle>']
    with open(output_path + "g_accuracies", 'rb') as f:
        accuracies = pickle.load(f)

    acc_y = [accuracies[k] for k in accuracies.keys()]
    pltutil.plot_single_stat(accuracies.keys(), acc_y, "Accuracy")

if __name__ == 'main':
  main() 