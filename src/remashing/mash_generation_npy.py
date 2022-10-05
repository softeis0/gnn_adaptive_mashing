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

        self.passive_refinement_1 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_2 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_3 = np.empty([1, 3], dtype=int)  # indices

        # changed triangles that are good and need to be refined.
        self.changed_triangles = np.empty([1, 3], dtype=int)   # indices

        #self.remashed_Graph = None

    def adaptive_refinement(self, max_error):
        self.init_low_high_error(max_error)
        while (self.triangles_high_Error.size != 0):

            print(self.triangles_high_Error.shape)
            node_index_new_nodes = self.nodes_numpy.shape[0]
            for triangle in self.triangles_high_Error:
                self.refine_one_bad_triangle(triangle)
            self.destroy_triangles(self.triangles_high_Error)
            self.changed_triangles = self.changed_triangles[1:]
            new_nodes = self.nodes_numpy[node_index_new_nodes:]
            nodes_set_created = set([tuple(t) for t in new_nodes.tolist()])

            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created)

            self.passive_refinement_1 = self.passive_refinement_1[1:]
            self.passive_refinement_2 = self.passive_refinement_2[1:]
            self.passive_refinement_3 = self.passive_refinement_3[1:]
            x = 0
            return

            self.sort_passive_refinement_cases()
            self.take_care_of_passive_refinement_2()
            for t in self.passive_refinement_3:
                self.take_care_of_passive_refinement_3(t)
            for t in self.passive_refinement_1:
                if t.is_special_triangle():
                    self.take_care_of_passive_refinement_1_special(t)
            for t in self.passive_refinement_1:
                self.take_care_of_passive_refinement_1(t)

            self.actualize_neighbors_t(list(self.created_triangles))
            self.triangles.extend(self.created_triangles)

            # clear sets that were used for one iteration
            self.triangles_high_Error.clear()
            self.sort_triangles_into_low_high_error(triangles=self.created_triangles,
                                                    max_error=max_error)
            # print(self.triangles_low_Error)
            self.created_triangles.clear()
            self.created_nodes.clear()
            self.changed_neighbors.clear()
            self.passive_refinement_1.clear()
            self.passive_refinement_2.clear()
            self.passive_refinement_3.clear()
            self.changed_triangles.clear()

        print("---")
        graph_nodes = set()
        # graph_nodes = graph_nodes & set(t.graph.x.numpy)
        self.remashed_Graph = Data()

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
        dif_sum = np.sum(difference)
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
        new_nodes_indices = self.add_nodes_global(nodes=new_nodes)
        indices = np.array((triangle, new_nodes_indices)).reshape(6,)

        # create new triangles using indices
        self.make_4_new_triangles(nodes=indices)

    def find_changed_triangles(self, triangle):
        x = np.equal(self.triangles_low_Error, triangle)
        y = np.sum(x, axis=1)
        z = 0

    """
 def find_changed_triangles(self, triangle, triangle_compare):
        if np.sum(np.equal(triangle, triangle_compare)) == 2:
            self.changed_triangles = np.vstack([self.changed_triangles, triangle])
    """
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
        self.nodes_numpy = np.vstack([self.nodes_numpy, nodes])
        new_indices_nodes = np.arange(self.nodes_numpy.shape[0] - 3, self.nodes_numpy.shape[0])
        return new_indices_nodes

    def make_4_new_triangles(self, nodes):
        triangle_1 = nodes[[0, 3, 4]]
        triangle_2 = nodes[[1, 3, 5]]
        triangle_3 = nodes[[2, 4, 5]]
        self.triangles_numpy = np.vstack([self.triangles_numpy, triangle_1, triangle_2, triangle_3])

    def destroy_triangles(self, triangles): #[tuple(node) for node in new_nodes]
        triangles_set = set([tuple(t) for t in self.triangles_numpy.tolist()])
        triagnles_set_old = set([tuple(t) for t in triangles.tolist()])
        self.triangles_numpy = np.array(list(triangles_set - triagnles_set_old))