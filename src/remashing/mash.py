from src.remashing.triangle import Triangle
import torch
import numpy as np
from torch_geometric.data import Data

class Mash:
    def __init__(self, graph):
        self.graph = graph
        self.triangles = [Triangle(graph.subgraph(torch.tensor([0, 1, 4])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([1, 2, 5])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([0, 3, 4])), 1, 1, 1),
                 Triangle(graph.subgraph(torch.tensor([1, 4, 5])), 1, 1, 1)]
        self.triangles_low_Error = []
        self.triangles_high_Error = []

        # link neighbor of triangles
        self.actualize_neighbors_t(self.triangles)

    def adaptive_remash_Graph(self):
        pass

    def adaptive_procedure(self):
        pass
        # return Graph, first part

    def adaptive_refinement(self, max_error, idx_feature):
        self.sort_triangles_into_low_high_error(self.triangles)
        while(self.triangles_high_Error):
            self.created_triangles = set()
            self.created_nodes = set()
            self.changed_triangles = set()

            for t in self.triangles_high_Error:
                new_triangles, new_nodes, self.changed_neighbors = self.refine_triangle_bad(t)
                self.created_triangles = (set(new_triangles) | self.created_triangles)
                nodes_tuple = [tuple(node) for node in new_nodes]
                self.created_nodes = (set(nodes_tuple) | self.created_nodes)
                self.changed_triangles = (set(self.changed_neighbors) | self.changed_triangles)

            good_changed_triangles = list(self.changed_triangles & set(self.triangles_low_Error))
            for t in good_changed_triangles:


    # input: list of triangles.
    # the function links all the neighbors of the input triangles as neighbors.
    def actualize_neighbors_t(self, triangles):
        triangles_copy = triangles.copy()
        for t in triangles:
            triangles_copy.remove(t)
            for t_compare in triangles_copy:
                if self.triangles_are_neighbors(t, t_compare):
                    t.add_neighbor(t_compare)

    # input: two triangles t1 and t2.
    # output: True if they are neighbors, else False.
    def triangles_are_neighbors(self, t1, t2):
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

    # gets triangles, the feature for the error and the max_error allowed.
    # sorts all input triangles into self.triangles_low_Error or self.triangles_high_Error.
    def sort_triangles_into_low_high_error(self, triangles, idx_feature, max_error):
        for t in triangles:
            print(t.get_error(idx_feature))
            if t.get_error(idx_feature) < max_error:
                self.triangles_low_Error.append(t)
            else:
                self.triangles_high_Error.append(t)

    # destroy old triangle and make 4 new ones, get the new triangles, nodes, and affected neighbors as output.
    def refine_triangle_bad(self, triangle):
        # gets all 6 nodes, the first three being the original ones.
        all_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())
        # store old_neighbors of triangle (they may need to be redefined)
        old_neighbors = triangle.neighbors

        # create four new triangles
        old_nv = [triangle.i1_nv, triangle.i2_nv, triangle.i3_nv]
        triangles = self.make_4_triangles(all_nodes, old_nv)

        # destroy old triangle
        triangle.remove_all_neighbors()
        self.triangles.remove(triangle)
        return triangles, all_nodes[-3:, :].tolist(), old_neighbors

    # gets the three old nodes (np_array) as input, output is a np_array with the old and new nodes.
    def add_3_new_nodes(self, old_nodes):
        result = old_nodes
        result = np.append(result, [(result[0, :] + result[1, :]) / 2], axis=0)
        result = np.append(result, [(result[0, :] + result[2, :]) / 2], axis=0)
        result = np.append(result, [(result[1, :] + result[2, :]) / 2], axis=0)
        return result

    # makes for triangles out of the 6 nodes (6a,c,d and 7a,b2) and returns them in a list.
    def make_4_triangles(self, nodes, old_nv):
        node_3_nv = max(old_nv[0], old_nv[1]) + 1
        node_4_nv = max(old_nv[0], old_nv[2]) + 1
        node_5_nv = max(old_nv[1], old_nv[2]) + 1

        return [Triangle(graph=self.make_triangle_graph(nodes[0, :], nodes[3, :], nodes[4, :]),
                         i1_nv=old_nv[0],
                         i2_nv=node_3_nv,
                         i3_nv=node_4_nv),
                Triangle(graph=self.make_triangle_graph(nodes[1, :], nodes[3, :], nodes[5, :]),
                         i1_nv=node_3_nv,
                         i2_nv=old_nv[2],
                         i3_nv=node_5_nv),
                Triangle(graph=self.make_triangle_graph(nodes[2, :], nodes[4, :], nodes[5, :]),
                         i1_nv=old_nv[2],
                         i2_nv=node_4_nv,
                         i3_nv=node_5_nv),
                Triangle(graph=self.make_triangle_graph(nodes[3, :], nodes[4, :], nodes[5, :]),
                         i1_nv=node_3_nv,
                         i2_nv=node_4_nv,
                         i3_nv=node_5_nv)]

    # input are three nodes, output graph of a triange.
    def make_triangle_graph(self, v1, v2, v3):
        x = torch.from_numpy(np.stack((v1, v2, v3)))
        edge_index = torch.tensor([[0, 1], [0, 2], [1, 2]])
        return Data(x=x, edge_index=edge_index)