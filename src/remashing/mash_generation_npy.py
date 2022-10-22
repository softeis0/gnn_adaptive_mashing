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
        self.triangle_index = [0]
        self.new_triangles = None

        self.passive_refinement_1 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_1_nodes = list()
        self.passive_refinement_triangles_from_1 = None
        self.passive_refinement_triangles_from_1_index = 0

        self.passive_refinement_2 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_2_nodes = list()
        self.passive_refinement_triangles_from_2 = None
        self.passive_refinement_triangles_from_2_index = [0]
        self.passive_refinement_nodes_from_2 = None
        self.passive_refinement_nodes_from_2_index = 0

        self.passive_refinement_3 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_3 = None
        self.passive_refinement_triangles_from_3_index = [0]


    def adaptive_refinement(self, max_error):
        self.init_low_high_error(max_error)
        i = 0
        while (self.triangles_high_Error.size != 0):

            self.new_nodes = np.full(fill_value=-1,shape=(self.triangles_high_Error.shape[0]*3, 3), dtype=float)
            self.new_triangles = np.full(fill_value=-1, shape=(self.triangles_high_Error.shape[0] * 4, 3), dtype=int)

            for triangle in self.triangles_high_Error:
                self.refine_one_bad_triangle(triangle)

            self.new_nodes = self.new_nodes[self.new_nodes != -1].reshape(-1, 3)
            self.nodes_numpy = np.vstack([self.nodes_numpy, self.new_nodes])
            self.new_nodes_index = 0
            self.triangles_numpy = np.vstack([self.triangles_numpy, self.new_triangles])
            array_index = 0

            self.destroy_triangles(self.triangles_high_Error)

            nodes_set_created = set([tuple(t) for t in self.new_nodes.tolist()])

            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created,insert_t1=False, insert_t2=True, insert_t3=False)
            self.passive_refinement_2 = self.passive_refinement_2[1:]

            while (self.passive_refinement_2.shape[0] > 0):
                self.insert_refinement_t_2()
                for triangle in self.triangles_low_Error:
                    self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created, insert_t1=False, insert_t2=True, insert_t3=False)
                self.passive_refinement_2 = self.passive_refinement_2[1:]

            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created,
                                                  insert_t1=True, insert_t2=False, insert_t3=True)

            self.passive_refinement_1 = self.passive_refinement_1[1:]
            self.passive_refinement_3 = self.passive_refinement_3[1:]


            self.insert_refinement_t_1()

            # self.passive_refinement_3 = np.vstack([self.passive_refinement_3, self.triangles_numpy[:5]])
            self.insert_refinement_t_3()

            self.prepare_next_loop()

            self.init_low_high_error(max_error)
            i += 1
        print(i)

    # set init values to prepare the next loop
    def prepare_next_loop(self):
        self.new_nodes = None
        self.new_nodes_index = 0
        self.new_triangles = None

        self.passive_refinement_1 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_1 = None
        self.passive_refinement_triangles_from_1_index = 0

        self.passive_refinement_2 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_2 = None
        self.passive_refinement_triangles_from_2_index = [0]
        self.passive_refinement_nodes_from_2 = None
        self.passive_refinement_nodes_from_2_index = 0

        self.passive_refinement_3 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_3 = None
        self.passive_refinement_triangles_from_3_index = [0]

        self.triangle_index = [0]

        self.triangles_low_Error = None  # indices
        self.triangles_high_Error = None  # indices
        self.triangles_low_Error_index = 0
        self.triangles_high_Error_index = 0

    # solve the passive refinement case 1
    def insert_refinement_t_1(self):
        local_copy_refinement_t_1 = self.passive_refinement_1
        self.passive_refinement_triangles_from_1 = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_1.shape[0] * 2, 3),
                                                           dtype=int)
        self.take_care_of_passive_refinement_1_npy()
        self.triangles_numpy = np.vstack([self.triangles_numpy, self.passive_refinement_triangles_from_1])
        self.destroy_triangles(local_copy_refinement_t_1, destroy_in_low_Error=True)

    # create new triangles for passive refinement case 1
    def take_care_of_passive_refinement_1_npy(self):
        while self.passive_refinement_1.shape[0] > 0:
            triangle, self.passive_refinement_1 = self.passive_refinement_1[-1], self.passive_refinement_1[:-1]
            new_node = self.passive_refinement_1_nodes.pop()

            old_nodes = self.get_triangle_nodes(triangle=triangle)
            possible_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

            # indice of the new node
            indice_new_node = np.where(np.equal(self.nodes_numpy, np.array(list(new_node))[0]).all(1) == True)[0][0]

            if np.array_equal(np.array(list(new_node))[0], possible_nodes[0, :]):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[0], triangle[2]]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[1], triangle[2]]
                self.passive_refinement_triangles_from_1_index += 1
            elif np.array_equal(np.array(list(new_node)), possible_nodes[1, :].reshape(-1, 3)):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[0], triangle[1]]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[2], triangle[1]]
                self.passive_refinement_triangles_from_1_index += 1
            elif np.array_equal(np.array(list(new_node)), possible_nodes[2, :].reshape(-1, 3)):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[2], triangle[0]]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[1], triangle[0]]
                self.passive_refinement_triangles_from_1_index += 1

    # solve the passive refinement case 2
    def insert_refinement_t_2(self):
        local_copy_refinement_t_2 = self.passive_refinement_2
        self.passive_refinement_triangles_from_2 = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_2.shape[0] * 4, 3),
                                                           dtype=int)
        self.passive_refinement_nodes_from_2 = np.full(fill_value=-1,
                                                       shape=(self.passive_refinement_2.shape[0], 3),
                                                       dtype=float)
        self.take_care_of_passive_refinement_2_npy()
        self.passive_refinement_nodes_from_2 = self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2 != -1].reshape(-1, 3)
        self.nodes_numpy = np.vstack([self.nodes_numpy, self.passive_refinement_nodes_from_2])
        self.triangles_numpy = np.vstack([self.triangles_numpy, self.passive_refinement_triangles_from_2])
        self.destroy_triangles(local_copy_refinement_t_2, destroy_in_low_Error=True)
        self.passive_refinement_2 = np.empty([1, 3], dtype=int)

    # create new triangles/nodes for passive refinement case 2
    def take_care_of_passive_refinement_2_npy(self):
        while self.passive_refinement_2.shape[0] > 0:

            triangle, self.passive_refinement_2 = self.passive_refinement_2[-1], self.passive_refinement_2[:-1]

            new_node, new_nodes = self.get_new_node(triangle=triangle, created_nodes=self.passive_refinement_2_nodes.pop())

            indices_new_nodes = self.add_nodes_global(nodes=new_nodes,ref_2=True)
            indices = np.array((triangle, indices_new_nodes)).reshape(6, )

            # create new triangles using indices
            self.make_4_new_triangles(nodes=indices, triangle_array=self.passive_refinement_triangles_from_2, array_index=self.passive_refinement_triangles_from_2_index)

    # get the one node that needs t be created for passive refinement case 2
    def get_new_node(self, triangle, created_nodes):
        old_nodes = self.get_triangle_nodes(triangle=triangle)
        new_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

        set_new_nodes = set([tuple(x) for x in list(new_nodes)])
        set_created_nodes = set(created_nodes)

        return set_new_nodes - (set_new_nodes & set_created_nodes), new_nodes

    # solve the passive refinement case 3
    def insert_refinement_t_3(self):
        local_copy_refinement_t_3 = self.passive_refinement_3
        self.passive_refinement_triangles_from_3 = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_3.shape[0] * 4, 3),
                                                           dtype=int)
        self.take_care_of_passive_refinement_3_npy()
        self.triangles_numpy = np.vstack([self.triangles_numpy, self.passive_refinement_triangles_from_3])
        self.destroy_triangles(local_copy_refinement_t_3, destroy_in_low_Error=True)

    # create new triangles for passive refinement case 3
    def take_care_of_passive_refinement_3_npy(self):
        while self.passive_refinement_3.shape[0] > 0:
            triangle, self.passive_refinement_3 = self.passive_refinement_3[-1], self.passive_refinement_3[:-1]

            old_nodes = self.get_triangle_nodes(triangle=triangle)
            new_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

            # get indices of all the nodes
            indices_new_nodes = self.add_nodes_global(nodes=new_nodes)
            indices = np.array((triangle, indices_new_nodes)).reshape(6, )

            # create new triangles using indices
            self.make_4_new_triangles(nodes=indices, triangle_array=self.passive_refinement_triangles_from_3, array_index=self.passive_refinement_triangles_from_3_index)

    # sort the triangles into the group where they are affected passively (1/2/3) and which are activated through insert_*
    def sort_triangles_into_affected(self, triangle, nodes_set_created, insert_t1=False, insert_t2=False, insert_t3=False):
        duplicated_nodes = self.get_duplicated_nodes(triangle, nodes_set_created)
        is_triangle_affected = len(duplicated_nodes)
        if is_triangle_affected == 3 and insert_t3:
            self.passive_refinement_3 = np.vstack([self.passive_refinement_3, triangle])
        elif is_triangle_affected == 2 and insert_t2:
            self.passive_refinement_2 = np.vstack([self.passive_refinement_2, triangle])
            self.passive_refinement_2_nodes.append(duplicated_nodes)
        elif is_triangle_affected == 1 and insert_t1:
            self.passive_refinement_1 = np.vstack([self.passive_refinement_1, triangle])
            self.passive_refinement_1_nodes.append(duplicated_nodes)

    # get the nodes on which the triangle is passively affected
    def get_duplicated_nodes(self, triangle, nodes_set_created):
        new_nodes_triangle = self.create_3_new_nodes(self.get_triangle_nodes(triangle=triangle))
        nodes_set_triangle = set([tuple(t) for t in new_nodes_triangle.tolist()])
        return nodes_set_triangle & nodes_set_created

    # TODO hier liegt 15% der Zeit
    # initialize triangles into high and low error
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

    # function to look if one triangle needs to go into high or low error
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

    # refines one bad triangle
    def refine_one_bad_triangle(self, triangle):
        # get old and new nodes
        old_nodes = self.get_triangle_nodes(triangle=triangle)
        new_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

        # get indices of all the nodes
        indices_new_nodes = self.add_nodes_global(nodes=new_nodes)
        indices = np.array((triangle, indices_new_nodes)).reshape(6,)

        # create new triangles using indices
        self.make_4_new_triangles(nodes=indices, triangle_array=self.new_triangles, array_index=self.triangle_index)


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

    # adds new nodes to nodes_result_indices. returns the new indices of the nodes. ref_2 changes the method for the refinement case 2
    # #TODO: hier liegt 85% der Zeit
    def add_nodes_global(self, nodes, ref_2=False):
        if ref_2:
            a = np.equal(self.passive_refinement_nodes_from_2, nodes[0]).all(1)
            if (a.any()):
                indice_0 = np.where(a == True)[0][0]
            else:
                b = np.equal(self.nodes_numpy, nodes[0]).all(1)
                if (b.any()):
                    indice_0 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[0]
                    indice_0 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
            a = np.equal(self.passive_refinement_nodes_from_2, nodes[1]).all(1)
            if (a.any()):
                indice_1 = np.where(a == True)[0][0]
            else:
                b = np.equal(self.nodes_numpy, nodes[1]).all(1)
                if (b.any()):
                    indice_1 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[1]
                    indice_1 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
                a = np.equal(self.passive_refinement_nodes_from_2, nodes[2]).all(1)
            if (a.any()):
                indice_2 = np.where(a == True)[0][0]
            else:
                b = np.equal(self.nodes_numpy, nodes[2]).all(1)
                if (b.any()):
                    indice_2 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[2]
                    indice_2 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
            return np.array([indice_0, indice_1, indice_2])
        else:
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

    # makes 4 new triangles and sorts them into the triangle array with the indices given
    def make_4_new_triangles(self, nodes, triangle_array, array_index):
        triangle_array[array_index, :] = nodes[[0, 3, 4]]
        array_index[0] += 1
        triangle_array[array_index, :] = nodes[[1, 3, 5]]
        array_index[0] += 1
        triangle_array[array_index, :] = nodes[[2, 4, 5]]
        array_index[0] += 1
        triangle_array[array_index, :] = nodes[[3, 4, 5]]
        array_index[0] += 1

    # destroys a certain amount of triangles for self.triangles_numpy. if destory in low_Error=True, triangles get taken out of self.triangles_low_Error too.
    def destroy_triangles(self, triangles, destroy_in_low_Error=False):
        triangles_set = set([tuple(t) for t in self.triangles_numpy.tolist()])
        triagnles_set_old = set([tuple(t) for t in triangles.tolist()])
        self.triangles_numpy = np.array(list(triangles_set - triagnles_set_old))
        if destroy_in_low_Error:
            low_error_triangles_set = set([tuple(t) for t in self.triangles_low_Error])
            self.triangles_low_Error = np.array(list(low_error_triangles_set - triagnles_set_old))