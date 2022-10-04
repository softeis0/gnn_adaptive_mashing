from src.remashing.triangle_mash import Triangle
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from tqdm import trange

class MashNpy:
    def __init__(self, graph):
        self.graph = graph
        self.triangles_numpy = graph['triangles'].numpy()
        self.nodes_numpy = graph['x'].numpy()
        self.edges_numpy = graph['edge_index'].numpy()
        self.triangles_low_Error = np.empty([1, 3], dtype=int) #indices
        self.triangles_high_Error = np.empty([1, 3], dtype=int) #indices

        self.passive_refinement_1 = np.zeros(0, dtype=int) #indices
        self.passive_refinement_2 = np.zeros(0, dtype=int) #indices
        self.passive_refinement_3 = np.zeros(0, dtype=int) #indices

        # triangles nad nodes created in this iteration
        self.created_triangles = np.zeros(0, dtype=int) #indices
        self.created_nodes = np.zeros(0, dtype=int) #indices
        # changed triangles that are good and need to be refined.
        self.changed_triangles = np.zeros(0, dtype=int) #indices

        self.remashed_Graph = None


    def adaptive_refinement(self, max_error):
        for i in trange(1):
            # sort triangles in high/low Error
            np.apply_along_axis(self.sort_triangles_into_low_high_error, 1, self.triangles_numpy, max_error)
            # remove initialized values.
            self.triangles_low_Error = self.triangles_low_Error[1:]
            self.triangles_high_Error = self.triangles_high_Error[1:]

            while(self.triangles_high_Error):

                for t in self.triangles_high_Error:
                    new_triangles, new_nodes, self.changed_neighbors = self.refine_triangle_bad(t)
                    self.created_triangles = (set(new_triangles) | self.created_triangles)
                    nodes_tuple = [tuple(node) for node in new_nodes]
                    self.created_nodes = (set(nodes_tuple) | self.created_nodes)
                    self.changed_triangles = ((set(self.changed_neighbors) & set(self.triangles_low_Error)) | self.changed_triangles)

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
                #print(self.triangles_low_Error)
                self.created_triangles.clear()
                self.created_nodes.clear()
                self.changed_neighbors.clear()
                self.passive_refinement_1.clear()
                self.passive_refinement_2.clear()
                self.passive_refinement_3.clear()
                self.changed_triangles.clear()


            print("---")
            graph_nodes = set()
                #graph_nodes = graph_nodes & set(t.graph.x.numpy)
            self.remashed_Graph = Data()

    def sort_triangles_into_low_high_error(self, triangle, max_error):
        # get nodes of triangle
        nodes = np.take(self.nodes_numpy, triangle, axis=0)
        nodes = np.vstack([nodes, nodes[0]])
        # calculate error
        difference = np.abs(np.diff(nodes, axis=0))
        dif_sum = np.sum(difference)
        if dif_sum < max_error:
            self.triangles_low_Error = np.vstack([self.triangles_low_Error, triangle])
        else:
            self.triangles_high_Error = np.vstack([self.triangles_high_Error, triangle])

