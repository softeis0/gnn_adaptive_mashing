import torch
from torch_geometric.data import Data
import numpy as np

class Triangle:

    def __init__(self, graph, i1_nv, i2_nv, i3_nv):
        # i1 i2 and i3 are tensors
        self.graph = graph
        self.i1_nv = i1_nv
        self.i2_nv = i2_nv
        self.i3_nv = i3_nv
        self.neighbors = []
        self.split_partner = None

    def get_error(self, constant=1):
        error = 0
        nodes =  self.graph.to_dict()['x']
        feature_amount = nodes[1,:].size()[0]
        for i in range(3, feature_amount):
            error += self.get_error_one_feature(idx_feature=i)
        return error

    def get_error_one_feature(self, idx_feature, constant=1):
        nodes = self.graph.to_dict()['x']
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
        neighbors_copy = self.neighbors.copy()
        for t in neighbors_copy:
            self.remove_neighbor(t)

    def is_special_triangle(self):
        nvs = [self.i1_nv, self.i2_nv, self.i3_nv]
        max_nv = max(nvs)
        count = 0
        for nv in nvs:
            if nv == max_nv:
                count += 1
        if count < 2:
            return True
        else:
            return False

    def set_split_partner(self, triangle):
        self.split_partner = triangle

    def to_numpy(self):
        return self.graph.x.numpy()
"""
    def mark_neighbors(self, all_nodes):
        for t in self.neighbors:
            t.mark(self, all_nodes)

    def mark(self, triangle, all_nodes):
        self.marked_neighbors.append(triangle)
        self.all_nodes = all_nodes
        """