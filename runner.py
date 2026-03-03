from algorithm import beam_search_path
from graph_util.make_adj_list import process_graphs
import os
import sys
import model.net as net
import pickle

sys.stdout = open("evaluation/approximate_paths.txt", "w")

base = os.path.dirname(__file__)              
path =  os.path.join(base, "datasets/graphs.txt")
graphs = process_graphs(path, 30)

base = os.path.dirname(__file__)             
pathh = os.path.join(base, "model/trained_model.pkl")
with open(pathh, "rb") as f:
    sys.modules['__main__'] = net
    model = pickle.load(f)

print(len(graphs))
for adj in graphs:
    best_path = beam_search_path(adj, model, 3)
    print(len(best_path))
    print(*best_path)

