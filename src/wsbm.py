import networkx as nx
import graph_tool.all as gt

import sklearn
from sklearn import metrics

from scipy.optimize import linear_sum_assignment

import math
import numpy as np
import random 

import pymc3 as pm

import sys

np.set_printoptions(threshold=sys.maxsize)

from collections import defaultdict


def make_weights(G=nx.Graph(), attributes=[], test_statistic=np.median, non_value=0.0, normalization=lambda x: x):
  """
  Update edge weights from annotation attributes.
  :param G: graph
  :param attributes: list of attributes to be summarized
  :param test_statistic: test statistic to summarize data
  :param non_value: value of non-judgment
  :param normalization: normalization function
  :return G: updated graph
  """
  for (i,j) in G.edges():

      #print(G[i][j])

      values = [G[i][j]['attributes'][attribute] for attribute in attributes if attribute in G[i][j]['attributes']]         
              
      if values == []:
          sys.exit('Breaking: No attribute matching edge: (%s,%s)' % (i,j))

      data = [v for v in values if not v == non_value] # exclude non-values

      if data!=[]:        
          weight = normalization(test_statistic(data))
      else:
          weight = float('nan')

      G[i][j]['weight'] = weight
  return G

def sem_eval_2_graph_tool(graph, pos):
  gt_true = gt.Graph(directed=False)

  weights = []
  
  s_vid = dict()
  count = 0

  for node in graph.nodes():
    s_vid[node] = count
    count += 1

  for i,j in graph.edges():
    current_weight = graph[i][j]['weight']
    if current_weight != 0 and not math.isnan(current_weight):
      
      if isinstance(current_weight, float):
        round_func = math.ceil if bool(random.getrandbits(1)) else math.floor
        current_weight = round_func(current_weight)
      
      gt_true.add_edge(s_vid[i], s_vid[j])
      weights.append(current_weight)

  ew = gt_true.new_edge_property("double")
  ew.a = weights 
  gt_true.ep['orig_weight'] = ew

  iw = gt_true.new_edge_property("int")
  iw.a = list(map(lambda x: int(x * 2 - 2), ew.a))
  gt_true.ep['shifted_weight'] = iw

  sw = gt_true.new_edge_property("double")
  sw.a = list(map(lambda x: x - 2.5, ew.a))

  gt_true.ep['weight'] = sw
  
  vid = gt_true.new_vertex_property("string")
  gt_true.vp.id = vid
  
  for k,v in s_vid.items():
    vid[v] = k
  
  vpos = gt_true.new_vertex_property("vector<double>")
  gt_true.vp.pos = vpos

  
  for key,val in pos.items():
    v = [v for v in gt_true.get_vertices() if gt_true.vp.id[gt_true.vertex(v)] == key][0]
    vpos[gt_true.vertex(v)] = val
  
  return gt_true


def open_graph(g_path):

  graph = nx.read_gpickle(g_path)

  graph = make_weights(
      graph, 
      attributes=['annotator1', 'annotator2', 'annotator3', 'annotator4', 'annotator5', 'annotator6', 'annotator7', 'annotator8', 'annotator9', 'annotator10', 'annotator11', 'annotator12', 'annotator13', 'annotator14', 'annotator15'], 
      non_value=0.0)
  
  G_pos = graph.copy()
  edges_negative = [(i,j) for (i,j) in G_pos.edges() if G_pos[i][j]['weight'] < 2.5]
  G_pos.remove_edges_from(edges_negative)
  
  pos = nx.nx_agraph.graphviz_layout(
      G_pos, 
      args='-LN<200>',
      prog='sfdp'
      )
  
  s_gt = sem_eval_2_graph_tool(graph, pos)

  return graph, s_gt, pos

def minimize(s_gt, distribution="discrete-binomial", pure_w=True, deg_corr=False, overlap=False):
  state = gt.minimize_blockmodel_dl(
      s_gt, 
      B_min=1, 
      B_max=30,
      deg_corr=deg_corr, 
      state_args=dict(recs=[s_gt.ep.shifted_weight],rec_types=[distribution]), 
      mcmc_args=dict(niter=100,entropy_args=dict(adjacency=False,degree_dl=False)),
      verbose=False,
      overlap=overlap)
      
  return state

def get_blocks(state):
  b = state.get_blocks()
  b = gt.perfect_prop_hash([b])[0]
  return b

def find_best_distribution(g, verbose=False):
  
  distributions = ["discrete-binomial", 
                  "discrete-geometric", 
                  "discrete-poisson"]
  
  best_distribution = ""
  best_state = None
  current_entropy = np.inf

  for dist in distributions:
      state = minimize(g, overlap=False, deg_corr=False)
      entropy = state.entropy(adjacency=False,degree_dl=False)
      if entropy < current_entropy:
        current_entropy = entropy
        best_distribution = dist
        best_state = state
  return best_distribution, best_state

def compute_partition(g, b):
  # partition graph
  edges = g.get_edges([g.ep.orig_weight])
  
  partition = defaultdict(list)

  for edge in edges:
    b1 = b[edge[0]]
    b2 = b[edge[1]]

    if b1 > b2:
      b1, b2 = b2, b1

    key = str(b1) if b1 == b2 else str(b1) + "-" + str(b2)

    partition[key].append(edge[2])

  return partition

def infer_p(values, distribution="discrete-binomial"):
  
  alpha = 0.1
  beta = 0.1

  with pm.Model() as model: # context management
    
    if distribution == "discrete-binomial":
      p = pm.Beta('p', alpha=alpha, beta=beta)
      y = pm.Binomial('y', n=4, p=p, observed=values)
    elif distribution == "discrete-geometric":
      p = pm.Beta('p', alpha=alpha, beta=beta)
      y = pm.Geometric('y', p=p, observed=values)
    elif distribution == "discrete-poisson":
      p = pm.Gamma('p', alpha=alpha, beta=beta)
      y = pm.Poisson('y', mu=p, observed=values)
    
    trace = pm.sample(2000, tune=1000, cores=4)

    print(pm.summary(trace).to_string())
    #pm.traceplot(trace)

    return trace['p'].mean()

def purity_score(y_true, y_pred):
  # compute contingency matrix (also called confusion matrix)
  contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
  # return purity
  return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def cluster_accuracy(y_true, y_pred):
  # compute contingency matrix (also called confusion matrix)
  contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

  # Find optimal one-to-one mapping between cluster labels and true labels
  row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

  # Return cluster accuracy
  return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def partition_and_stat_gt(graph, s_gt, b, verbose=False):
  c_s = defaultdict(list)
  c_g = defaultdict(list)

  l_g = []
  l_s = []

  for node in graph.nodes():
    l_s.append(graph.nodes[node]['cluster'])
    v = [v for v in s_gt.vertices() if s_gt.vp.id[v] == node][0]
    l_g.append(b[v])
    c_s[int(graph.nodes[node]['cluster'])].append(node)

  for node in s_gt.vertices():
    c_g[b[node]].append(s_gt.vp.id[node])


  mri = sklearn.metrics.adjusted_rand_score(l_s, l_g)

  purity = purity_score(l_s, l_g)
  acc = cluster_accuracy(l_s, l_g)

  if verbose:
    print("purity: " + str(purity))
    print("accuracy: " + str(acc))
    print("Mean Rand Index: " + str(mri))
    print("Clusters from Sem_Eval Graph: " + str(set(l_s)))
    print("Clusters inferred: " + str(set(l_g)))
  

  return mri, purity, acc