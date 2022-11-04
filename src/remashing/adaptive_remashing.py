# algorithm for adaptive remashing
import torch
from torch_geometric.data import Data
from torch_geometric import utils as geom_utils
from src.remashing.triangle_mash import Triangle
from src.remashing.mash import Mash
from src.remashing.mash_generation_npy import MashNpy
import cProfile
import numpy as np
from src.tests.visualize_mash import plot_sphere

def euclidean_distance(x, y):
    sq_dist = torch.sum((x-y)**2, dim=1)
    return torch.sqrt(sq_dist)

def construct_pygraph(points, values, triangles):

    hull = triangles
    edge_index = np.vstack([hull[:, [0, 1]], hull[:, [0, 2]], hull[:, [1, 2]]])
    edge_index = np.unique(edge_index, axis=0)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)

    vertex_attributes = torch.tensor(np.concatenate([points, values], axis=-1), dtype=torch.float32)

    edge_length = euclidean_distance(vertex_attributes[edge_index[0], :3], vertex_attributes[edge_index[1], :3])

    return Data(x=vertex_attributes, edge_index=edge_index, edge_attr=edge_length)

def main():
    path_path = '../data/graph.pt'
    path_basegraph = '../data/basegraph.pt'
    mash = MashNpy(graph=torch.load(path_path), basegraph=torch.load(path_basegraph))
    mash.adaptive_refinement(max_error=1.5)
    mash.triangles_numpy = mash.triangles_numpy[mash.triangles_numpy[:,0].argsort()]
    points = mash.nodes_numpy[:, :3]
    values = mash.nodes_numpy[:, -2:]
    plot_sphere(points=[points[:, 0], points[:, 1], points[:, 2]], values=values[:, 0])
    pygraph = construct_pygraph(points=points, values=values, triangles=mash.triangles_numpy)
    # test_mash = Mash(make_test_graph3())
    # test_mash.adaptive_refinement(max_error=4)

if __name__=="__main__":
    cProfile.run('main()', "output.dat")
    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()

    with open("output_cumtime.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("cumtime").print_stats()


#test_adaptive_refinement(graph=graph, max_error=0.1, idx_feature=3)


