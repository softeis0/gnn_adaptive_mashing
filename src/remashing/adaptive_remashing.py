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

def test_adaptive_refinement(graph, max_error, idx_feature):
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
        created_triangles = set()
        created_nodes = set()
        changed_triangles = set()

        for t in t_bad:
            new_triangles, new_nodes, changed_neighbors = refine_triangle_bad(t)
            created_triangles = (set(new_triangles) | created_triangles)
            nodes_tuple = [tuple(node) for node in new_nodes]
            created_nodes = (set(nodes_tuple) | created_nodes)
            changed_triangles = (set(changed_neighbors) | changed_triangles)

        good_changed_triangles = list(changed_triangles & set(t_good))
        for t in good_changed_triangles:
            created_triangles = refine_triangle_good(t, created_nodes)


def refine_triangle_good(triangle, created_nodes):
    changed_nodes = get_changed_nodes(triangle, created_nodes)[2:,:]
    if len(changed_nodes) == 3:
        new_triangles, new_nodes = refine_passive_3_node(triangle=triangle, new_nodes=changed_nodes)
    elif len(changed_nodes) == 2:
        refine_passive_2_node()
    elif len(changed_nodes) == 1:
        refine_passive_1_node()
    else:
        print("Error, len(changed_nodes) muss zwischen 1 und 3 sein.")

def get_changed_nodes(triangle, created_nodes):
    possible_nodes = get_3_new_nodes(triangle.graph.x.numpy())
    return possible_nodes & created_nodes

def refine_passive_1_node(triangle, new_nodes):
    pass

def refine_passive_2_node(triangle, new_nodes):
    pass

def refine_passive_3_node(triangle, new_nodes):
    all_nodes = get_3_new_nodes(triangle.graph.x.numpy())
    old_neighbors = triangle.neighbors
    old_nv = [triangle.i1_nv, triangle.i2_nv, triangle.i3_nv]
    triangle.delete_triangle()
    triangles = make_4_triangles(all_nodes, old_nv)
    return triangles, all_nodes[-3:, :].tolist(), old_neighbors

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
    triangle.remove_all_neighbors()
    triangles = make_4_triangles(all_nodes, old_nv)
    return triangles, all_nodes[-3:,:].tolist(), old_neighbors

def get_3_new_nodes(x):
    result = x
    result = np.append(result, [(result[0, :] + result[1, :]) / 2], axis=0)
    result = np.append(result, [(result[0, :] + result[2, :]) / 2], axis=0)
    result = np.append(result, [(result[1, :] + result[2, :]) / 2], axis=0)
    return result

def make_4_triangles(nodes, old_nv):
    node_3_nv = max(old_nv[0], old_nv[1]) + 1
    node_4_nv = max(old_nv[0], old_nv[2]) + 1
    node_5_nv = max(old_nv[1], old_nv[2]) + 1


    return [Triangle(graph=make_triangle_graph(nodes[0, :], nodes[3, :], nodes[4, :]),
                     i1_nv=old_nv[0],
                     i2_nv=node_3_nv,
                     i3_nv=node_4_nv),
                 Triangle(graph=make_triangle_graph(nodes[1, :], nodes[3, :], nodes[5, :]),
                          i1_nv=node_3_nv,
                          i2_nv=old_nv[2],
                          i3_nv=node_5_nv),
                 Triangle(graph=make_triangle_graph(nodes[2, :], nodes[4, :], nodes[5, :]),
                          i1_nv=old_nv[2],
                          i2_nv=node_4_nv,
                          i3_nv=node_5_nv),
                 Triangle(graph=make_triangle_graph(nodes[3, :], nodes[4, :], nodes[5, :]),
                          i1_nv=node_3_nv,
                          i2_nv=node_4_nv,
                          i3_nv=node_5_nv)]

def make_triangle_graph(v1, v2, v3):
    x = torch.from_numpy(np.stack((v1, v2, v3)))
    edge_index = torch.tensor([[0,1], [0,2], [1,2]])
    return Data(x=x, edge_index=edge_index)

path = '../data/basegraph.pt'
mash = Mash(graph=torch.load(path))

#test_adaptive_refinement(graph=graph, max_error=0.1, idx_feature=3)
#test_mash = Mash(make_test_graph3())

#test_mash.adaptive_refinement(max_error=4)