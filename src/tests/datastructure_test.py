import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from torch_geometric import utils as geom_utils
import pymesh

path = '../data/basegraph.pt'
graph = torch.load(path)
triangles_numpy = graph['triangles']
nodes_numpy = graph['x']
edges_numpy = graph['edge_index']
mesh = pymesh.form_mesh(nodes_numpy, edges_numpy)
print(mesh)

# make own graph
x = torch.tensor([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 3, 3, 9, 1], [2, 1, 1, 1, 0.5], [2, 2, 2 , 7, 1], [2, 3, 3, 16, 1.5]])
edge_index =  torch.tensor([[0,0,0,1,1,1,2,3,4,1,3,4,2,4,5,5,4,5], [1,3,4,2,4,5,5,4,5,0,0,0,1,1,1,2,3,4]])
test_graph = Data(x=x, edge_index=edge_index)

# look at own graph
#print(test_graph)
#print(test_graph.is_directed())
#print(graph.get_edge_index())
#print(graph.put_edge_index( ).is_directed())

edge_attr = graph.edge_attr.tolist()

y = 0