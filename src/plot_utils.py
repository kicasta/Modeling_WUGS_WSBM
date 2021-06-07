import graph_tool.all as gt

import matplotlib.pyplot as plt

import scipy
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)
import PIL.Image

def combine_community_plots(imgs, community_count, title, path=""):

  blank_image = None

  for img_name in imgs:
    img = PIL.Image.open(path + img_name + ".png")

    if(len(img_name) == 1):
      x = int(img_name) * img.width
      y = int(img_name) * img.height
    else:
      parts = img_name.split('-')
      x = int(parts[0]) * img.width
      y = int(parts[1]) * img.height

    if blank_image == None:
      w = community_count * img.width
      h = community_count * img.height

      blank_image = PIL.Image.new("RGB", (w, h), color=(255, 255, 255))


    blank_image.paste(img, (x,y))
 
  blank_image.save(path + title)

def compute_sampled_true_labels(true_graph, true_labels, subgraph):
  sampled_true_labels = []
  for v in subgraph.vertices():
    i = 0
    for tv in true_graph.vertices():
      if subgraph.vp.id[v] == true_graph.vp.id[tv]:
        sampled_true_labels.append(true_labels[i])
        break
      i += 1
  return sampled_true_labels

def plot_values(inside, outside, inferred_inside_p, inferred_outside_p,title, path="", xticks_shifted = True, distribution="discrete-binomial"):
  fig, ax = plt.subplots(figsize=(14,7), squeeze=True)

  color_inside_observed = (0.411, 0.674, 0.909, 0.9)
  color_outside_observed = (0.933, 0.588, 0.623, 0.8)
  color_inside_inferred = (0.027, 0.419, 0.772)
  color_outside_inferred = (0.866, 0.031, 0.117)

  x, bins, p = plt.hist(
      inside, 
      density=True, 
      bins=[0,0.5,1.5,2.5,3.5,4.5],
      color=color_inside_observed,
      label="Inside Blocks Observed"
      )
  
  for item in p:
    item.set_height(item.get_height()/sum(x))

  if len(outside) > 0:
    x, bins, p = plt.hist(
        outside, 
        density=True, 
        bins=[0,0.5,1.5,2.5,3.5,4.5],
        color=color_outside_observed,
        label="Between Blocks Observed"
        )
    
    for item in p:
      item.set_height(item.get_height()/sum(x))
  
  x = np.arange(1,5)
  

  if distribution == "discrete-binomial":
    pmf = scipy.stats.binom.pmf(x, 4, inferred_inside_p)
  elif distribution == "discrete-poisson":
    pmf = scipy.stats.poisson.pmf(x, inferred_inside_p)
  elif distribution == "discrete-geometric":
    pmf = scipy.stats.geom.pmf(x, inferred_inside_p)
  inside_plot = plt.plot(x, pmf, color=color_inside_inferred, label="Inside Block Inferred")

  if len(outside) > 0:
    if distribution == "discrete-binomial":
      pmf = scipy.stats.binom.pmf(x, 4, inferred_outside_p)
    elif distribution == "discrete-poisson":
      pmf = scipy.stats.poisson.pmf(x, inferred_outside_p)
    elif distribution == "discrete-geometric":
      pmf = scipy.stats.geom.pmf(x, inferred_outside_p)
    outside_plot = plt.plot(x, pmf, color=color_outside_inferred, label="Outside Block Inferred")

  if xticks_shifted:
    locs, l = plt.xticks()
    labels = [str((x + 2) * 0.5 - 2.5) for x in locs]

    ax.set_xticklabels(labels)

  plt.legend(fontsize=20)
  plt.ylabel('Probability density', fontsize=22)
  plt.xlabel('Weights', fontsize=22)

  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)

  plt.savefig(fname=path + title, dpi=150)


def plot_values_oneside(observed_v, inferred_p, in_community, title, fig_title, path="", xticks_shifted = True, distribution="discrete-binomial"):
  fig, ax = plt.subplots(figsize=(14,7))
  color_inside_observed = (0.411, 0.674, 0.909, 0.9)
  color_outside_observed = (0.933, 0.588, 0.623, 0.8)
  color_inside_inferred = (0.027, 0.419, 0.772)
  color_outside_inferred = (0.866, 0.031, 0.117)

  prelabel = title

  x, bins, p = plt.hist(
      observed_v, 
      density=True, 
      bins=[0,0.5,1.5,2.5,3.5,4.5], 
      color=color_inside_observed if in_community else color_outside_observed,
      label= "Observed"
      )
  
  for item in p:
    item.set_height(item.get_height()/sum(x))
  
  x = np.arange(1,5)

  if distribution == "discrete-binomial":
    pmf = scipy.stats.binom.pmf(x, 4, inferred_p)
  elif distribution == "discrete-poisson":
    pmf = scipy.stats.poisson.pmf(x, inferred_p)
  elif distribution == "discrete-geometric":
    pmf = scipy.stats.geom.pmf(x, inferred_p)
  plt.plot(x, pmf, color=color_inside_inferred if in_community else color_outside_inferred, label="Inferred")

  if xticks_shifted:
    locs, l = plt.xticks()
    labels = [str((x + 2) * 0.5 - 2.5) for x in locs]

    ax.set_xticklabels(labels)

  ax.set_title(title, fontsize=22)

  plt.legend(fontsize=20)
  plt.ylabel('Probability density', fontsize=22)
  plt.xlabel('Weights', fontsize=22)

  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)
  
  plt.savefig(fname=path + fig_title, dpi=150)
  plt.close(fig)
  
def plot_dist_dist(labels, dists):

  y, x = (list(t) for t in zip(*sorted(zip(dists, labels), reverse=True)))

  fig, ax = plt.subplots(figsize=(5, 3))
  
  colors=['red', 'green', 'blue']

  plt.bar(x, y, color=colors, width=0.4)
 
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  ax.yaxis.label.set_size(18) 
  
  plt.ylabel("Number", fontsize=18)
  plt.show()

def draw(g, title, b=None, show_img=False):

  weight = g.ep["weight"]


   # color vectors
  gray=[0.5, 0.5, 0.5, 1.0]
  black=[0.1, 0.1, 0.1, 1.0]

  # edge color
  ecolor=g.new_edge_property("vector<double>")
  epen=g.new_edge_property("double")

  #print(g.ep.weight.a)

  for e in g.edges(): 
    if g.ep.orig_weight[e] == 4:
      ecolor[e] = black
      epen[e] = 2
    elif g.ep.orig_weight[e] == 3:
      ecolor[e] = black
      epen[e] = 0.9
    else:
      ecolor[e]= gray
      epen[e] = 0.5

  if show_img:
    gt.graph_draw(
      g, 
      pos=g.vp.pos, 
      vertex_size=12, 
      vertex_fill_color=b, 
      edge_color=ecolor, 
      edge_pen_width=epen, 
      fit_view=True,
      adjust_aspect=False,
      ink_scale=0.9,
      bg_color=[1,1,1,1]
      )
    
  else:
    gt.graph_draw(
      g, 
      pos=g.vp.pos, 
      vertex_size=12, 
      vertex_fill_color=b, 
      edge_color=ecolor, 
      edge_pen_width=epen, 
      output_size = (640, 480),
      fit_view=True,
      adjust_aspect=False,
      #output_size = (3840, 2160), #4k
      output=title,
      ink_scale=0.9,
      overlap=True
      )


def correct_utf8(graphs):
  utf8_correct = {"Fu":"Fuß", 
                "vergnnen": "vergönnen", 
                "Naturschnheit": "Naturschönheit",
                "abgebrht" : "abgebrüht",
                "Tragfhigkeit" : "Tragfähigkeit",
                "Miklang": "Mißklang",
                "Ackergert": "Ackergerät",
                "berspannen":"überspannen",
                "Engpa":"Engpaß"
                }
  labels = []
  for g in graphs:
    if g in utf8_correct:
        labels.append(utf8_correct[g])
    else:
      labels.append(g)
  return labels

def plot_single_stat(graphs, stat, label, limy=True):
  
  graphs = correct_utf8(graphs)
  
  fig, ax = plt.subplots(figsize=(14,7))

  plt.grid()
  zipped_pairs = zip(stat, graphs) 
  
  sorted_pairs = sorted(zipped_pairs, reverse=True)
  sorted_graphs = [x for _, x in sorted_pairs]
  sorted_stats = [x for x, _ in sorted_pairs]

  plt.scatter(sorted_graphs, sorted_stats)

  plt.ylabel(label, fontsize=18)

  plt.xticks(rotation=90)
  plt.yticks(fontsize=18)

  if limy:
    plt.ylim(0,1.1)
  else:
    plt.yticks(np.arange(min(stat), max(stat)+1, step=1.0))

  #plt.legend(fontsize=18)
  plt.show()
