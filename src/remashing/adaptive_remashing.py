# algorithm for adaptive remashing
import torch
from torch_geometric.data import Data
from torch_geometric import utils as geom_utils
from src.remashing.triangle import Triangle
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
        [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 3, 3, 9, 1], [2, 1, 1, 1, 0.5], [2, 2, 2, 7, 1], [2, 3, 3, 16, 1.5]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 4],
                               [1, 3, 4, 2, 4, 5, 5, 4, 5,]])
    return Data(x=x, edge_index=edge_index)

def test_adaptive_refinement(graph, max_error, idx_feature):
    x = 0
    # get nodes from graph in nodes
    #dict = graph.to_dict()
    #nodes = dict['x']

    # create all 4 triangles
    triangles = [Triangle(graph.subgraph(torch.tensor([0, 1, 4])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([1, 2, 5])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([0, 3, 4])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([1, 4, 5])), 1, 1, 1)]
    actualize_neighbors_t(triangles)

    # all triangles with an error less than max error are in t_good, the rest in t_bad
    t_good = []
    t_bad = []

    # sort triangles in good and bad
    for t in triangles:
        print(t.get_error(idx_feature))
        if t.get_error(idx_feature) < max_error:
            t_good.append(t)
        else:
            t_bad.append(t)

    # until all triangles have a error less max_error
    while t_bad:
        for t in t_bad:
            refine_triangle_bad(t)

def actualize_neighbors_t(triangles):
    triangles_copy = triangles.copy()
    for t in triangles:
        triangles_copy.remove(t)
        for t_compare in triangles_copy:
            if triangles_are_neighbors(t, t_compare):
                t.add_neighbor(t_compare)


def triangles_are_neighbors(t1, t2):
    equal_nodes = 0
    x1 = t1.graph.x
    x2 = t2.graph.x
    # check how many nodes are equal.
    for x11 in x1:
        for x22 in x2:
            if torch.equal(x22, x11):
                equal_nodes += 1
    # decide if triangles are neighbors, based on equal_nodes
    if equal_nodes == 2:
        return True
    elif equal_nodes > 2:
        print("Error, mehr als 2 nodes gleich bei Dreieckvergleich")
    else:
        return False

def refine_triangle_bad(triangle):
    # the first 3 are original, then combinations.
    all_nodes = get_3_new_nodes(triangle.graph.x.numpy())
    old_neighbors = triangle.neighbors
    old_nv = [triangle.i1_nv, triangle.i2_nv, triangle.i3_nv]
    triangle.delete_triangle()
    triangles = make_4_triangles(all_nodes, old_nv)
    y = 0

def get_3_new_nodes(x):
    result = x
    result = np.append(result, [(result[0, :] + result[1, :]) / 2], axis=0)
    result = np.append(result, [(result[0, :] + result[2, :]) / 2], axis=0)
    result = np.append(result, [(result[1, :] + result[2, :]) / 2], axis=0)
    return result

def make_4_triangles(nodes, old_nv):
    return [Triangle(graph=make_triangle_graph(nodes[0, :], nodes[1, :], nodes[3, :]),
                     i1_nv=old_nv[0],
                     i2_nv=old_nv[1],
                     i3_nv=max(old_nv[0], old_nv[1]) + 1),
                 Triangle(graph=make_triangle_graph(nodes[0, :], nodes[2, :], nodes[4, :]),
                          i1_nv=old_nv[0],
                          i2_nv=old_nv[2],
                          i3_nv=max(old_nv[0], old_nv[2]) + 1),
                 Triangle(graph=make_triangle_graph(nodes[1, :], nodes[2, :], nodes[5, :]),
                          i1_nv=old_nv[1],
                          i2_nv=old_nv[2],
                          i3_nv=max(old_nv[1], old_nv[2]) + 1),
                 Triangle(graph=make_triangle_graph(nodes[3, :], nodes[4, :], nodes[5, :]),
                          i1_nv=max(old_nv[0], old_nv[1]) + 1,
                          i2_nv=max(old_nv[0], old_nv[2]) + 1,
                          i3_nv=max(old_nv[1], old_nv[2]) + 1)]

def make_triangle_graph(v1, v2, v3):
    x = torch.tensor([[v1], [v2], [v3]])
    edge_index = torch.tensor([[0,1], [0,2], [1,2]])
    return Data(x=x, edge_index=edge_index)

test_adaptive_refinement(graph=make_test_graph(), max_error=1, idx_feature=3)