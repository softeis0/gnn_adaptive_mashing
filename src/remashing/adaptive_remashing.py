# algorithm for adaptive remashing
import torch
from torch_geometric.data import Data
from torch_geometric import utils as geom_utils
from src.remashing.triangle import Triangle
from src.remashing.mash import Mash
import numpy as np


def adaptive_remash_Graph(Graph):
    pass

def adaptive_procedure(Graph):
    pass
    # return Graph, first part

def adaptive_refinement(Graph):
    pass
    # return Graph, second part

def make_test_graph():
    x = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 2, 1, 2, 2, 1], [1, 3, 1, 3, 9, 1], [2, 1, 1, 1, 1, 0.5], [2, 2, 1, 2, 7, 1], [2, 3, 1, 3, 16, 1.5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4],
                               [1, 3, 4, 2, 4, 5, 5, 4, 5,]])
    return Data(x=x, edge_index=edge_index)

def make_test_graph2():
    x = torch.tensor(
        [[1, 1, 1, 16, 93, -33], [1, 2, 1, 3, 15, -7], [1, 3, 1, 40, 92, 10], [2, 1, 1, -33, -33, 12], [2, 2, 1, 0, -12, -20], [2, 3, 1, -20, 16, 30]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4],
                               [1, 3, 4, 2, 4, 5, 5, 4, 5,]])
    return Data(x=x, edge_index=edge_index)

def make_test_graph3():
    x = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 2, 1, 0, 1, 1], [1, 3, 1, 3, 9, 1], [2, 1, 1, 20, 15, 10], [2, 2, 1, 2, 1, 1], [2, 3, 1, 9, 16, 25]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4],
                               [1, 3, 4, 2, 4, 5, 5, 4, 5,]])
    return Data(x=x, edge_index=edge_index)



path = '../data/basegraph.pt'
mash = Mash(graph=torch.load(path))

mash.adaptive_refinement(max_error=10)
#test_adaptive_refinement(graph=graph, max_error=0.1, idx_feature=3)
#test_mash = Mash(make_test_graph3())

#test_mash.adaptive_refinement(max_error=4)