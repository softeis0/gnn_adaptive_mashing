from src.remashing.triangle_mash import Triangle
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from tqdm import trange
import profile

class MashNpy:
    def __init__(self, graph):
        self.graph = graph
        self.triangles_numpy = graph['triangles'].numpy()
        self.nodes_numpy = graph['x'].numpy()
        self.edges_numpy = graph['edge_index'].numpy()

        self.triangles_low_Error = None  # indices
        self.triangles_high_Error = None  # indices
        self.triangles_low_Error_index = 0
        self.triangles_high_Error_index = 0

        self.new_nodes_index = 0
        self.new_nodes = None
        self.new_triangles_index = 0
        self.new_triangles = None

        self.passive_refinement_1 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_2 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_3 = np.empty([1, 3], dtype=int)  # indices


    def adaptive_refinement(self, max_error):
        self.init_low_high_error(max_error)
        while (self.triangles_high_Error.size != 0):

            self.new_nodes = np.full(fill_value=-1,shape=(self.triangles_high_Error.shape[0]*3, 3), dtype=float)
            self.new_triangles = np.full(fill_value=-1, shape=(self.triangles_high_Error.shape[0] * 4, 3), dtype=float)

            for triangle in self.triangles_high_Error:
                self.refine_one_bad_triangle(triangle)

            self.new_nodes = self.new_nodes[self.new_nodes != -1].reshape(-1, 3)
            self.nodes_numpy = np.vstack([self.nodes_numpy, self.new_nodes])
            self.new_nodes_index = 0
            self.triangles_numpy = np.vstack([self.triangles_numpy, self.new_triangles])
            self.new_triangles_index = 0

            self.destroy_triangles(self.triangles_high_Error)

            nodes_set_created = set([tuple(t) for t in self.new_nodes.tolist()])

            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created)

            self.passive_refinement_1 = self.passive_refinement_1[1:]
            self.passive_refinement_2 = self.passive_refinement_2[1:]
            self.passive_refinement_3 = self.passive_refinement_3[1:]
            x = 0
            return

            self.take_care_of_passive_refinement_2()
            for t in self.passive_refinement_3:
                self.take_care_of_passive_refinement_3(t)
            for t in self.passive_refinement_1:
                if t.is_special_triangle():
                    self.take_care_of_passive_refinement_1_special(t)
            for t in self.passive_refinement_1:
                self.take_care_of_passive_refinement_1(t)


            # clear sets that were used for one iteration
            self.triangles_high_Error.clear()
            self.sort_triangles_into_low_high_error(triangles=self.created_triangles,
                                                    max_error=max_error)

            self.passive_refinement_1.clear()
            self.passive_refinement_2.clear()
            self.passive_refinement_3.clear()


    def take_care_of_passive_refinement_2(self):
        while (self.passive_refinement_2):
            t = self.passive_refinement_2.pop()
            new_node = self.get_third_node_passive_refinement_2(t)
            self.created_nodes = self.created_nodes | set([tuple(x) for x in new_node.tolist()])
            self.passive_refinement_3.append(t)
            t_affected = self.triangle_affected_by_new_node(t.neighbors, new_node)
            if t_affected:
                if t_affected in self.passive_refinement_1:
                    self.passive_refinement_1.remove(t_affected)
                    self.passive_refinement_2.append(t_affected)
                elif t_affected in self.passive_refinement_2:
                    self.passive_refinement_2.remove(t_affected)
                    self.passive_refinement_3.append(t_affected)
                else:
                    self.passive_refinement_1.append(t_affected)

    def get_third_node_passive_refinement_2(self, triangle):
        possible_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())[3:, :]
        possible_nodes_set = set([tuple(x) for x in possible_nodes])
        result = np.asarray(list(possible_nodes_set - self.created_nodes))
        return result

    def take_care_of_passive_refinement_2_npy(self):
        pass



    def sort_triangles_into_affected(self, triangle, nodes_set_created):
        new_nodes_triangle = self.create_3_new_nodes(self.get_triangle_nodes(triangle=triangle))

        nodes_set_triangle = set([tuple(t) for t in new_nodes_triangle.tolist()])

        is_triangle_affected = len(nodes_set_triangle & nodes_set_created)
        if is_triangle_affected == 3:
            self.passive_refinement_3 = np.vstack([self.passive_refinement_3, triangle])
        elif is_triangle_affected == 2:
            self.passive_refinement_2 = np.vstack([self.passive_refinement_2, triangle])
        elif is_triangle_affected == 1:
            self.passive_refinement_1 = np.vstack([self.passive_refinement_1, triangle])

    def init_low_high_error(self, max_error):
        self.triangles_low_Error_index = 0
        self.triangles_high_Error_index = 0
        self.triangles_low_Error = np.full(fill_value=-1,shape=self.triangles_numpy.shape)  # indices
        self.triangles_high_Error = np.full(fill_value=-1,shape=self.triangles_numpy.shape)  # indices
        # sort triangles in high/low Error
        np.apply_along_axis(self.sort_triangles_into_low_high_error, 1, self.triangles_numpy, max_error)
        # remove initialized values.
        self.triangles_low_Error = self.triangles_low_Error[self.triangles_low_Error != -1].reshape(-1, 3)
        self.triangles_high_Error = self.triangles_high_Error[self.triangles_high_Error != -1].reshape(-1, 3)

    def sort_triangles_into_low_high_error(self, triangle, max_error):
        # get nodes of triangle
        triangle_nodes = self.get_triangle_nodes(triangle=triangle)
        nodes = np.zeros(shape=(4,3))
        nodes[0] = triangle_nodes[0]
        nodes[1] = triangle_nodes[1]
        nodes[2] = triangle_nodes[2]
        nodes[3] = triangle_nodes[0]
        # calculate error
        difference = np.abs(np.diff(nodes, axis=0))
        dif_sum = difference.sum()
        if dif_sum < max_error:
            self.triangles_low_Error[self.triangles_low_Error_index, :] = triangle
            self.triangles_low_Error_index += 1
        else:
            self.triangles_high_Error[self.triangles_high_Error_index, :] = triangle
            self.triangles_high_Error_index += 1

    def refine_one_bad_triangle(self, triangle):
        # get old and new nodes
        old_nodes = self.get_triangle_nodes(triangle=triangle)
        new_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

        # get indices of all the nodes
        indices_new_nodes = self.add_nodes_global(nodes=new_nodes)
        indices = np.array((triangle, indices_new_nodes)).reshape(6,)

        # create new triangles using indices
        self.make_4_new_triangles(nodes=indices)


    # return 3 new nodes out of an old
    def create_3_new_nodes(self, old_nodes):
        node1 = (old_nodes[0, :] + old_nodes[1, :]) / 2
        node2 = (old_nodes[0, :] + old_nodes[2, :]) / 2
        node3 = (old_nodes[1, :] + old_nodes[2, :]) / 2
        return np.array((node1, node2, node3))

    # get the nodes of a triangle
    def get_triangle_nodes(self, triangle):
        nodes = self.nodes_numpy[triangle]
        # nodes = np.take(self.nodes_numpy, triangle, axis=0)
        return nodes

    # adds new nodes to nodes_numpy + nodes_result_indices. returns the new indices of the nodes
    def add_nodes_global(self, nodes):

        a = np.equal(self.new_nodes, nodes[0]).all(1)
        if (a.any()):
            indice_0 = np.where(a == True)[0][0]
        else:
            self.new_nodes[self.new_nodes_index, :] = nodes[0]
            indice_0 = self.new_nodes_index
            self.new_nodes_index += 1
        a = np.equal(self.new_nodes, nodes[1]).all(1)
        if (a.any()):
            indice_1 = np.where(a == True)[0][0]
        else:
            self.new_nodes[self.new_nodes_index, :] = nodes[1]
            indice_1 = self.new_nodes_index
            self.new_nodes_index += 1
        a = np.equal(self.new_nodes, nodes[2]).all(1)
        if (a.any()):
            indice_2 = np.where(a == True)[0][0]
        else:
            self.new_nodes[self.new_nodes_index, :] = nodes[2]
            indice_2 = self.new_nodes_index
            self.new_nodes_index += 1
        return np.array([indice_0, indice_1, indice_2]) + self.nodes_numpy.shape[0]

    def make_4_new_triangles(self, nodes):
        self.new_triangles[self.new_triangles_index, :] = nodes[[0, 3, 4]]
        self.new_triangles_index += 1
        self.new_triangles[self.new_triangles_index, :] = nodes[[1, 3, 5]]
        self.new_triangles_index += 1
        self.new_triangles[self.new_triangles_index, :] = nodes[[2, 4, 5]]
        self.new_triangles_index += 1
        self.new_triangles[self.new_triangles_index, :] = nodes[[3, 4, 5]]
        self.new_triangles_index += 1

    def destroy_triangles(self, triangles):
        triangles_set = set([tuple(t) for t in self.triangles_numpy.tolist()])
        triagnles_set_old = set([tuple(t) for t in triangles.tolist()])
        self.triangles_numpy = np.array(list(triangles_set - triagnles_set_old))