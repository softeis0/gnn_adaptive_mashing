import torch
from torch_geometric.data import Data

class Triangle:

    def __init__(self, graph, i1_nv, i2_nv, i3_nv):
        # i1 i2 and i3 are tensors
        self.graph = graph
        self.i1_nv = i1_nv
        self.i2_nv = i2_nv
        self.i3_nv = i3_nv
        self.neighbors = []

    def get_error(self, idx_feature, constant=1):
        nodes =  self.graph.to_dict()['x']
        return max(abs(nodes[0, idx_feature] - nodes[1, idx_feature]),
                   abs(nodes[1, idx_feature] - nodes[2, idx_feature]),
                   abs(nodes[2, idx_feature] - nodes[0, idx_feature])).item() * constant

    # gets triangle as input. If the triangle isn't linked as neighbor with self, they get linked.
    def add_neighbor(self, triangle):
        if not (triangle in self.neighbors):
            self.neighbors.append(triangle)
            triangle.add_neighbor(self)

    # unlinks self with input triangle. (if they are neighbors)
    def remove_neighbor(self, triangle):
        if (triangle in self.neighbors):
            self.neighbors.remove(triangle)
            triangle.remove_neighbor(self)

    # unlinks self with all neighbor triangles.
    def remove_all_neighbors(self):
        for t in self.neighbors:
            t.remove_neighbor(self)