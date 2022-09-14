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

        self.passive_refinement_1 = []
        self.passive_refinement_2 = []
        self.passive_refinement_3 = []

        # triangles nad nodes created in this iteration
        self.created_triangles = set()
        self.created_nodes = set()
        # changed triangles that are good and need to be refined.
        self.changed_triangles = set()

    def adaptive_remash_Graph(self):
        pass

    def adaptive_procedure(self):
        pass
        # return Graph, first part

    def adaptive_refinement(self, max_error):
        self.sort_triangles_into_low_high_error(triangles=self.triangles,
                                                max_error=max_error)
        while(self.triangles_high_Error):

            for t in self.triangles_high_Error:
                new_triangles, new_nodes, self.changed_neighbors = self.refine_triangle_bad(t)
                self.created_triangles = (set(new_triangles) | self.created_triangles)
                nodes_tuple = [tuple(node) for node in new_nodes]
                self.created_nodes = (set(nodes_tuple) | self.created_nodes)
                self.changed_triangles = ((set(self.changed_neighbors)& set(self.triangles_low_Error)) | self.changed_triangles)

            self.sort_passive_refinement_cases()
            self.take_care_of_passive_refinement_2()
            for t in self.passive_refinement_3:
                self.take_care_of_passive_refinement_3(t)
            for t in self.passive_refinement_1:
                self.take_care_of_passive_refinement_1(t)

            self.actualize_neighbors_t(self.created_triangles)
            self.triangles.extend(self.created_triangles)

            # clear sets that were used for one iteration
            self.triangles_high_Error.clear()
            self.sort_triangles_into_low_high_error(triangles=self.created_triangles,
                                                    max_error=max_error)
            print(self.triangles_low_Error)
            self.created_triangles.clear()
            self.created_nodes.clear()
            self.changed_neighbors.clear()


        print("---")
        for t in self.triangles:
            print(t.get_error())


    def take_care_of_passive_refinement_1(self, triangle):
        new_node = self.get_changed_nodes(triangle=triangle)
        possible_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())
        if np.array_equal(new_node, possible_nodes[3, :]):
            two_triangles, two_neighbors = self.refine_triangle_1(triangle=triangle,
                                                                  new_node=new_node,
                                                                  shared_node=possible_nodes[2, :],
                                                                  node_3=possible_nodes[1, :],
                                                                  node_4=possible_nodes[0, :],
                                                                  shared_node_nv=triangle.i3_nv,
                                                                  node_3_nv=triangle.i2_nv,
                                                                  node_4_nv=triangle.i1_nv,
                                                                  new_node_nv=max(triangle.i1_nv, triangle.i2_nv) + 1)
        self.actualize_neighbors_t(two_triangles.extend(two_neighbors))
        self.created_triangles = self.created_triangles | set(two_triangles)

    def refine_triangle_1(self, triangle,  new_node, shared_node, node_3, node_4, shared_node_nv, node_3_nv, node_4_nv, new_node_nv):
        old_neighbors = triangle.neighbors
        triangles = [Triangle(graph=self.make_triangle_graph(new_node, shared_node, node_3),
                     i1_nv=new_node_nv,
                     i2_nv=shared_node_nv,
                     i3_nv=node_3_nv),
                     Triangle(graph=self.make_triangle_graph(new_node, shared_node, node_4),
                              i1_nv=new_node_nv,
                              i2_nv=shared_node_nv,
                              i3_nv=node_4_nv)]
        new_neighbors = [n for n in old_neighbors if self.triangles_are_neighbors(triangles[0], triangle) or self.triangles_are_neighbors(triangles[1], triangle)]
        # destory old triangle
        self.triangles_low_Error.remove(triangle)
        triangle.remove_all_neighbors()
        self.triangles.remove(triangle)

        return triangles, new_neighbors

    # take care of the passive refinement triangles case 3
    def take_care_of_passive_refinement_3(self, triangle):
        # gets all 6 nodes, the first three being the original ones.
        all_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())

        # create four new triangles
        old_nv = [triangle.i1_nv, triangle.i2_nv, triangle.i3_nv]
        triangles = self.make_4_triangles(all_nodes, old_nv)
        self.created_triangles.extend(triangles)

        # destroy old triangle
        self.triangles_low_Error.remove(triangle)
        triangle.remove_all_neighbors()
        self.triangles.remove(triangle)

    # take care of the passive refinement triangles case 2 and get them into the third ore first refinement case
    def take_care_of_passive_refinement_2(self):
        while (self.passive_refinement_2):
            t = self.passive_refinement_2.pop()
            new_node = self.get_third_node_passive_refinement_2(t)
            self.created_nodes.append(new_node)
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

    # get all the passive refinement triangles into the three different categories
    def sort_passive_refinement_cases(self):
        for t in self.changed_triangles:
            changed_nodes = self.get_changed_nodes(t)[3:, :]
            if len(changed_nodes) == 3:
                self.passive_refinement_3.append(t)
            elif len(changed_nodes) == 2:
                self.passive_refinement_3.append(t)
            elif len(changed_nodes) == 1:
                self.passive_refinement_3.append(t)
            else:
                print("Error, len(changed_nodes) muss zwischen 1 und 3 sein.")

    # returns the third node
    def get_third_node_passive_refinement_2(self, triangle):
        possible_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())[3:, :]
        return (possible_nodes - self.created_nodes)

    # returns the nodes that change the triangle
    def get_changed_nodes(self, triangle):
        possible_nodes = self.add_3_new_nodes(triangle.graph.x.numpy())[3:,:]
        x = np.asarray(self.created_nodes)
        return possible_nodes & x

    # returns the triangle that is affected by the new node.
    def triangle_affected_by_new_node(self, triangles, new_node):
        possible_nodes = self.add_3_new_nodes(triangles.graph.x.numpy())[3:,:]
        for t in triangles:
            if new_node in possible_nodes:
                return t
        return None


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
    def sort_triangles_into_low_high_error(self, triangles, max_error):
        for t in triangles:
            print(t.get_error())
            if t.get_error() < max_error:
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