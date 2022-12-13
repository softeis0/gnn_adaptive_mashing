import numpy as np
from src.tests.visualize_mash import plot_sphere
import pyvista as pv

class MashNpy:
    def __init__(self, graph, basegraph):
        self.graph = graph
        self.triangles_numpy = graph['triangles'].numpy()
        self.features = graph['x'].numpy()
        self.coordinates = basegraph['x'].numpy()
        self.nodes_numpy = np.hstack([self.coordinates, self.features])
        self.edges_numpy = graph['edge_index'].numpy()
        self.amount_features = self.features.shape[1]
        self.length_nodes = self.nodes_numpy.shape[1]
        self.triangles_Error = None

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

        self.nodes_set_changed = False
        self.passive_refinement_2 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_2_nodes = list()
        self.passive_refinement_triangles_from_2 = None
        self.passive_refinement_triangles_from_2_index = [0]
        self.passive_refinement_nodes_from_2 = None
        self.passive_refinement_nodes_from_2_index = 0

        self.passive_refinement_3 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_3 = None
        self.passive_refinement_triangles_from_3_index = [0]

        #self.show_mash()

        self.triangles_special_t1 = np.full(fill_value=-1,shape=(1, 9), dtype=int)  # indices
        self.triangles_special_t1_new = None
        self.triangles_special_t1_new_index = 0
        self.triangles_special_t1_to_destroy = None

    def adaptive_refinement(self, max_error):


        self.init_low_high_error(max_error)
        i = 0
        while (self.triangles_high_Error.size != 0):
            self.show_mash()
            self.new_nodes = np.full(fill_value=-1,shape=(self.triangles_high_Error.shape[0]*3, self.length_nodes), dtype=float)
            self.new_triangles = np.full(fill_value=-1, shape=(self.triangles_high_Error.shape[0] * 4, 3), dtype=int)

            for triangle in self.triangles_high_Error:
                self.refine_one_bad_triangle(triangle)

            self.new_nodes = self.new_nodes[self.new_nodes != -1].reshape(-1, self.length_nodes)
            self.nodes_numpy = np.vstack([self.nodes_numpy, self.new_nodes])
            self.new_nodes_index = 0
            self.triangles_numpy = np.vstack([self.triangles_numpy, self.new_triangles])

            self.destroy_triangles(self.triangles_high_Error)

            nodes_set_created = set([tuple(t) for t in self.new_nodes.round(5)])
            nodes_set_created_old = set([tuple(t) for t in self.nodes_numpy.round(5)])
            all_nodes_set = nodes_set_created_old | nodes_set_created

            first_loop = True
            new_refinements = False

            while(new_refinements | first_loop ):
                first_loop = False
                if np.array_equal(self.triangles_special_t1[0], np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1])):
                    continue
                self.triangles_special_t1_refine = np.full(fill_value=-1, shape=(self.triangles_special_t1.shape[0] * 4, 9),dtype=int)
                self.triangles_special_t1_refine_index = 0
                for special_triangle in self.triangles_special_t1:
                    self.sort_special_triangle_into_affected(special_triangle=special_triangle, nodes_set_created=nodes_set_created)

                self.triangles_special_t1_refine = self.triangles_special_t1_refine[self.triangles_special_t1_refine != -1].reshape(-1, 9)

                self.new_nodes_special_t1_nodes = np.full(fill_value=-1, shape=(self.triangles_special_t1_refine.shape[0]*2, self.length_nodes),dtype=float)
                self.triangles_special_t1_nodes_index = 0
                self.triangles_special_t1_new_triangles = np.full(fill_value=-1, shape=(
                self.triangles_special_t1_refine.shape[0] * 8, 3), dtype=int)
                self.triangles_special_t1_new_triangles_index = [0]
                self.triangles_special_t1_to_destroy = np.full(fill_value=-1, shape=(self.triangles_special_t1_refine.shape[0] * 2, 3))
                self.triangles_special_t1_to_destroy_index = 0

                for t in self.triangles_special_t1_refine:
                    self.solve_special_triangle_case(t, all_nodes_set)

                self.new_nodes_special_t1_nodes = self.new_nodes_special_t1_nodes[self.new_nodes_special_t1_nodes != -1].reshape(-1, self.length_nodes)
                self.triangles_special_t1_new_triangles = self.triangles_special_t1_new_triangles[self.triangles_special_t1_new_triangles != -1].reshape(-1, 3)
                self.nodes_numpy = np.vstack([self.nodes_numpy, self.new_nodes_special_t1_nodes])
                self.new_nodes = np.vstack([self.new_nodes, self.new_nodes_special_t1_nodes])
                self.triangles_numpy = np.vstack([self.triangles_numpy, self.triangles_special_t1_new_triangles])
                self.triangles_low_Error = np.vstack([self.triangles_low_Error, self.triangles_special_t1_new_triangles])
                new_refinements = self.triangles_special_t1_refine.shape[0] > 0

                self.triangles_special_t1_to_destroy = self.triangles_special_t1_to_destroy[
                    self.triangles_special_t1_to_destroy != -1].reshape(-1, 3)
                self.destroy_triangles(self.triangles_special_t1_to_destroy, destroy_in_low_Error = True)

            nodes_set_created = set([tuple(t) for t in self.new_nodes.round(5)])

            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created,insert_t1=False, insert_t2=True, insert_t3=False)
            self.passive_refinement_2 = self.passive_refinement_2[1:]

            while (self.passive_refinement_2.shape[0] > 0 or self.nodes_set_changed):
                old_nodes_set_created_len = len(nodes_set_created)
                self.insert_refinement_t_2()

                for triangle in self.triangles_low_Error:
                    self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created, insert_t1=False, insert_t2=True, insert_t3=False)
                nodes_set_created = set([tuple(t) for t in self.new_nodes.round(5)])
                self.passive_refinement_2 = self.passive_refinement_2[1:]
                self.nodes_set_changed = not (len(nodes_set_created) == old_nodes_set_created_len)


            for triangle in self.triangles_low_Error:
                self.sort_triangles_into_affected(triangle=triangle, nodes_set_created=nodes_set_created,
                                                  insert_t1=True, insert_t2=False, insert_t3=True)

            self.passive_refinement_1 = self.passive_refinement_1[1:]
            self.passive_refinement_3 = self.passive_refinement_3[1:]


            self.insert_refinement_t_1()
            if (self.triangles_special_t1.shape[0] > 1 and np.array_equal(self.triangles_special_t1[0], np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1]))):
                self.triangles_special_t1 = self.triangles_special_t1[1:]
            # self.passive_refinement_3 = np.vstack([self.passive_refinement_3, self.triangles_numpy[:5]])
            self.insert_refinement_t_3()

            self.prepare_next_loop()

            self.init_low_high_error(max_error)
            i += 1
            print(i)
        self.update_triangles_Error()


    def solve_special_triangle_case(self, special_triangle, all_nodes_set):
        self.triangles_special_t1_to_destroy[self.triangles_special_t1_to_destroy_index] = special_triangle[-9:-6]
        self.triangles_special_t1_to_destroy_index += 1
        self.triangles_special_t1_to_destroy[self.triangles_special_t1_to_destroy_index] = special_triangle[-6:-3]
        self.triangles_special_t1_to_destroy_index += 1


        triangle = special_triangle[-3:]

        new_node, new_nodes = self.get_new_node(triangle=triangle, created_nodes=all_nodes_set)

        indices_new_nodes = self.add_nodes_global_st(nodes=new_nodes)
        indices = np.array((triangle, indices_new_nodes)).reshape(6, )

        # create new triangles using indices
        self.make_4_new_triangles(nodes=indices, triangle_array=self.triangles_special_t1_new_triangles,
                                  array_index=self.triangles_special_t1_new_triangles_index)

    def add_nodes_global_st(self,nodes):
        a = np.isclose(self.nodes_numpy, nodes[0], atol=5e-05, rtol=0).all(1)
        if (a.any()):
            indice_0 = np.where(a == True)[0][0]
        else:
            self.new_nodes_special_t1_nodes[self.triangles_special_t1_nodes_index, :] = nodes[0]
            indice_0 = self.triangles_special_t1_nodes_index + self.nodes_numpy.shape[0]
            self.triangles_special_t1_nodes_index += 1
        a = np.isclose(self.nodes_numpy, nodes[1], atol=5e-05, rtol=0).all(1)
        if (a.any()):
            indice_1 = np.where(a == True)[0][0]
        else:
            self.new_nodes_special_t1_nodes[self.triangles_special_t1_nodes_index, :] = nodes[1]
            indice_1 = self.triangles_special_t1_nodes_index + self.nodes_numpy.shape[0]
            self.triangles_special_t1_nodes_index += 1
        a = np.isclose(self.nodes_numpy, nodes[2], atol=5e-05, rtol=0).all(1)
        if (a.any()):
            indice_2 = np.where(a == True)[0][0]
        else:
            self.new_nodes_special_t1_nodes[self.triangles_special_t1_nodes_index, :] = nodes[2]
            indice_2 = self.triangles_special_t1_nodes_index + self.nodes_numpy.shape[0]
            self.triangles_special_t1_nodes_index += 1
        return np.array([indice_0, indice_1, indice_2])

    def sort_special_triangle_into_affected(self, special_triangle, nodes_set_created):
        duplicated_nodes_1 = self.get_duplicated_nodes(special_triangle[-9:-6], nodes_set_created)
        duplicated_nodes_2 = self.get_duplicated_nodes(special_triangle[-6:-3], nodes_set_created)
        is_triangle_affected = len(duplicated_nodes_1) + len(duplicated_nodes_2)
        if is_triangle_affected > 1:
            self.triangles_special_t1_refine[self.triangles_special_t1_refine_index ] = special_triangle
            self.triangles_special_t1_refine_index += 1
            # destroy from special_triangle
            triangles_set = set([tuple(t) for t in self.triangles_special_t1.round(5)])
            triagnles_set_old = set([tuple(special_triangle.tolist())])
            self.triangles_special_t1 = np.array(list(triangles_set - triagnles_set_old))
        return

    def show_mash(self, triangle_error=True):

        points, values, triangles = self.nodes_numpy[:, :3], self.nodes_numpy[:, -2:], self.triangles_numpy
        # show_example()
        triangles_new = np.hstack([np.full(fill_value=3, shape=(triangles.shape[0], 1)), triangles])

        mesh = pv.PolyData(points, triangles_new)
        if triangle_error:
            self.update_triangles_Error()
            mesh.cell_data['Error'] = self.triangles_Error
        else:
            mesh.point_data['feature_1'] = values[:, 0]

        # Error von einem feature
        pl = pv.Plotter()
        point_labels = values[:, 0]
        pl.add_mesh(mesh, show_edges=True)
        # pl.add_point_labels(points, point_labels)
        pl.show()

    def update_triangles_Error(self):
        self.triangles_Error_index = 0
        self.triangles_Error = np.full(fill_value=-1, shape=(self.triangles_numpy.shape[0], 1), dtype=float)
        for triangle in self.triangles_numpy:
            self.get_error_for_triangle(triangle)

    def get_error_for_triangle(self, triangle):
        triangle_nodes = self.get_triangle_nodes(triangle=triangle)
        nodes = np.zeros(shape=(4, self.amount_features))
        nodes[0] = triangle_nodes[0, -self.amount_features:]
        nodes[1] = triangle_nodes[1, -self.amount_features:]
        nodes[2] = triangle_nodes[2, -self.amount_features:]
        nodes[3] = triangle_nodes[0, -self.amount_features:]
        # calculate error
        difference = np.abs(np.diff(nodes, axis=0))
        dif_sum = difference.sum()
        self.triangles_Error[self.triangles_Error_index] = dif_sum
        self.triangles_Error_index += 1

    # TODO hier liegt 15% der Zeit
    # initialize triangles into high and low error
    def init_low_high_error(self, max_error):
        self.triangles_low_Error_index = 0
        self.triangles_high_Error_index = 0
        self.triangles_low_Error = np.full(fill_value=-1, shape=self.triangles_numpy.shape)  # indices
        self.triangles_high_Error = np.full(fill_value=-1, shape=self.triangles_numpy.shape)  # indices
        # sort triangles in high/low Error
        np.apply_along_axis(self.sort_triangles_into_low_high_error, 1, self.triangles_numpy, max_error)
        # remove initialized values.
        self.triangles_low_Error = self.triangles_low_Error[self.triangles_low_Error != -1].reshape(-1, 3)
        self.triangles_high_Error = self.triangles_high_Error[self.triangles_high_Error != -1].reshape(-1, 3)

    # function to look if one triangle needs to go into high or low error
    def sort_triangles_into_low_high_error(self, triangle, max_error):
        # get nodes of triangle
        triangle_nodes = self.get_triangle_nodes(triangle=triangle)
        nodes = np.zeros(shape=(4, self.amount_features))
        nodes[0] = triangle_nodes[0, -self.amount_features:]
        nodes[1] = triangle_nodes[1, -self.amount_features:]
        nodes[2] = triangle_nodes[2, -self.amount_features:]
        nodes[3] = triangle_nodes[0, -self.amount_features:]
        # calculate error
        difference = np.abs(np.diff(nodes, axis=0))
        dif_sum = difference.sum()
        if (dif_sum / self.amount_features) < max_error:
            self.triangles_low_Error[self.triangles_low_Error_index, :] = triangle
            self.triangles_low_Error_index += 1
        else:
            self.triangles_high_Error[self.triangles_high_Error_index, :] = triangle
            self.triangles_high_Error_index += 1

    # set init values to prepare the next loop
    def prepare_next_loop(self):
        self.new_nodes = None
        self.new_nodes_index = 0
        self.new_triangles = None

        self.passive_refinement_1 = np.empty([1, 3], dtype=int)  # indices
        self.passive_refinement_triangles_from_1 = None
        self.passive_refinement_triangles_from_1_index = 0

        self.nodes_set_changed = False
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

        self.triangles_special_t1_new = None
        self.triangles_special_t1_new_index = 0

    # solve the passive refinement case 1
    def insert_refinement_t_1(self):
        local_copy_refinement_t_1 = self.passive_refinement_1
        self.passive_refinement_triangles_from_1 = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_1.shape[0] * 2, 3),
                                                           dtype=int)
        self.triangles_special_t1_new = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_1.shape[0] * 1, 9),
                                                           dtype=int)
        self.take_care_of_passive_refinement_1_npy()
        self.triangles_numpy = np.vstack([self.triangles_numpy, self.passive_refinement_triangles_from_1])
        self.triangles_special_t1 = np.vstack([self.triangles_special_t1, self.triangles_special_t1_new])

        self.destroy_triangles(local_copy_refinement_t_1, destroy_in_low_Error=True)

    # create new triangles for passive refinement case 1
    def take_care_of_passive_refinement_1_npy(self):
        while self.passive_refinement_1.shape[0] > 0:
            triangle, self.passive_refinement_1 = self.passive_refinement_1[-1], self.passive_refinement_1[:-1]
            new_node = self.passive_refinement_1_nodes.pop()

            old_nodes = self.get_triangle_nodes(triangle=triangle)
            possible_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

            self.triangles_special_t1_new[self.triangles_special_t1_new_index, -3:] = triangle

            # indice of the new node
            indice_new_node = np.where(np.isclose(self.nodes_numpy, np.array(list(new_node))[0], atol=5e-05, rtol=0).all(1) == True)[0][0]

            if np.allclose(np.array(list(new_node)), possible_nodes[0, :].reshape(-1, self.length_nodes), atol=5e-05, rtol=0):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[0], triangle[2]]

                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -6:-3] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[1], triangle[2]]
                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -9:-6] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1

            elif np.allclose(np.array(list(new_node)), possible_nodes[1, :].reshape(-1, self.length_nodes), atol=5e-05, rtol=0):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[0], triangle[1]]
                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -6:-3] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[2], triangle[1]]
                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -9:-6] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1
            elif np.allclose(np.array(list(new_node)), possible_nodes[2, :].reshape(-1, self.length_nodes), atol=5e-05, rtol=0):
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[2], triangle[0]]
                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -6:-3] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1
                self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index] = [
                    indice_new_node, triangle[1], triangle[0]]
                self.triangles_special_t1_new[self.triangles_special_t1_new_index, -9:-6] = self.passive_refinement_triangles_from_1[self.passive_refinement_triangles_from_1_index]
                self.passive_refinement_triangles_from_1_index += 1
            self.triangles_special_t1_new_index += 1

    # solve the passive refinement case 2
    def insert_refinement_t_2(self):
        self.passive_refinement_nodes_from_2_index = 0
        self.passive_refinement_triangles_from_2_index = [0]
        local_copy_refinement_t_2 = self.passive_refinement_2
        self.passive_refinement_triangles_from_2 = np.full(fill_value=-1,
                                                           shape=(self.passive_refinement_2.shape[0] * 4, 3),
                                                           dtype=int)
        self.passive_refinement_nodes_from_2 = np.full(fill_value=-1,
                                                       shape=(self.passive_refinement_2.shape[0], self.length_nodes),
                                                       dtype=float)
        self.take_care_of_passive_refinement_2_npy()
        self.passive_refinement_nodes_from_2 = self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2 != -1].reshape(-1, self.length_nodes)
        self.nodes_numpy = np.vstack([self.nodes_numpy, self.passive_refinement_nodes_from_2])
        self.new_nodes = np.vstack([self.new_nodes, self.passive_refinement_nodes_from_2])
        self.triangles_numpy = np.vstack([self.triangles_numpy, self.passive_refinement_triangles_from_2])
        self.destroy_triangles(local_copy_refinement_t_2, destroy_in_low_Error=True)
        self.passive_refinement_2 = np.empty([1, 3], dtype=int)

    # create new triangles/nodes for passive refinement case 2
    def take_care_of_passive_refinement_2_npy(self):
        while self.passive_refinement_2.shape[0] > 0:

            triangle, self.passive_refinement_2 = self.passive_refinement_2[-1], self.passive_refinement_2[:-1]

            new_node, new_nodes = self.get_new_node(triangle=triangle, created_nodes=self.passive_refinement_2_nodes[len(self.passive_refinement_2_nodes) - 1])

            indices_new_nodes = self.add_nodes_global(nodes=new_nodes,ref_2=True)
            indices = np.array((triangle, indices_new_nodes)).reshape(6, )

            # create new triangles using indices
            self.make_4_new_triangles(nodes=indices, triangle_array=self.passive_refinement_triangles_from_2, array_index=self.passive_refinement_triangles_from_2_index)

            self.passive_refinement_2_nodes.pop()
    # get the one node that needs t be created for passive refinement case 2
    def get_new_node(self, triangle, created_nodes):
        old_nodes = self.get_triangle_nodes(triangle=triangle)
        new_nodes = self.create_3_new_nodes(old_nodes=old_nodes)

        set_new_nodes = set([tuple(t) for t in new_nodes.round(5)])
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
            indices_new_nodes = self.add_nodes_global(nodes=new_nodes, ref_3=True)
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
        nodes_set_triangle = set([tuple(t) for t in new_nodes_triangle.round(5)])
        return nodes_set_triangle & nodes_set_created

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
        a = np.array([old_nodes[0, :] + old_nodes[1, :], old_nodes[0, :] + old_nodes[2, :], old_nodes[1, :] + old_nodes[2, :]])
        #b = np.array([, ,])
        """node1 = (old_nodes[0, :] + old_nodes[1, :]) / 2
        node2 = (old_nodes[0, :] + old_nodes[2, :]) / 2
        node3 = (old_nodes[1, :] + old_nodes[2, :]) / 2
        res2 = np.array((node1, node2, node3))"""
        return (a) / 2

    # get the nodes of a triangle
    def get_triangle_nodes(self, triangle):
        nodes = self.nodes_numpy[triangle]
        # nodes = np.take(self.nodes_numpy, triangle, axis=0)
        return nodes

    # adds new nodes to nodes_result_indices. returns the new indices of the nodes. ref_2 changes the method for the refinement case 2
    # #TODO: hier liegt 85% der Zeit
    def add_nodes_global(self, nodes, ref_2=False, ref_3=False):
        if ref_2:
            a = np.isclose(self.passive_refinement_nodes_from_2, nodes[0], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_0 = np.where(a == True)[0][0] + self.nodes_numpy.shape[0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[0], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_0 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[0]
                    indice_0 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
            a = np.isclose(self.passive_refinement_nodes_from_2, nodes[1], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_1 = np.where(a == True)[0][0] + self.nodes_numpy.shape[0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[1], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_1 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[1]
                    indice_1 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
            a = np.isclose(self.passive_refinement_nodes_from_2, nodes[2], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_2 = np.where(a == True)[0][0] + self.nodes_numpy.shape[0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[2], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_2 = np.where(b == True)[0][0]
                else:
                    self.passive_refinement_nodes_from_2[self.passive_refinement_nodes_from_2_index, :] = nodes[2]
                    indice_2 = self.passive_refinement_nodes_from_2_index + self.nodes_numpy.shape[0]
                    self.passive_refinement_nodes_from_2_index += 1
            return np.array([indice_0, indice_1, indice_2])
        elif ref_3:
            a = np.isclose(self.nodes_numpy, nodes[0], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_0 = np.where(a == True)[0][0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[0], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_0 = np.where(b == True)[0][0]
                else:
                    print("error rt3")
                    indice_0 = -1
            a = np.isclose(self.nodes_numpy, nodes[1], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_1 = np.where(a == True)[0][0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[1], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_1 = np.where(b == True)[0][0]
                else:
                    print("error rt3")
                    indice_1 = -1
            a = np.isclose(self.nodes_numpy, nodes[2], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_2 = np.where(a == True)[0][0]
            else:
                b = np.isclose(self.nodes_numpy, nodes[2], atol=5e-05, rtol=0).all(1)
                if (b.any()):
                    indice_2 = np.where(b == True)[0][0]
                else:
                    print("error rt3")
                    indice_2 = -1
            return np.array([indice_0, indice_1, indice_2])
        else:
            a = np.isclose(self.new_nodes, nodes[0], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_0 = np.where(a == True)[0][0]
            else:
                self.new_nodes[self.new_nodes_index, :] = nodes[0]
                indice_0 = self.new_nodes_index
                self.new_nodes_index += 1
            a = np.isclose(self.new_nodes, nodes[1], atol=5e-05, rtol=0).all(1)
            if (a.any()):
                indice_1 = np.where(a == True)[0][0]
            else:
                self.new_nodes[self.new_nodes_index, :] = nodes[1]
                indice_1 = self.new_nodes_index
                self.new_nodes_index += 1
            a = np.isclose(self.new_nodes, nodes[2], atol=5e-05, rtol=0).all(1)
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
    def destroy_triangles(self, triangles, destroy_in_low_Error=False, traingles_set=None, low_error_triangles_set=None):
        if triangles.size == 0:
            return
        triangles_set = set([tuple(t) for t in self.triangles_numpy.tolist()])
        triagnles_set_old = set([tuple(t) for t in triangles.tolist()])

        self.triangles_numpy = np.array(list(triangles_set - triagnles_set_old))
        if destroy_in_low_Error:
            low_error_triangles_set = set([tuple(t) for t in self.triangles_low_Error])
            self.triangles_low_Error = np.array(list(low_error_triangles_set - triagnles_set_old))
