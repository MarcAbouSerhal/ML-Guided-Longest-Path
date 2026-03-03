from graph_util.make_adj_list import process_graphs
import os
import sys
import copy

sys.stdout = open("evaluation/greedy_paths.txt", "w")

base = os.path.dirname(__file__)              
path =  os.path.join(base, "datasets/graphs.txt")
graphs = process_graphs(path, 30)

def find_greedy_paths(global_adj):
    results = []
    n = len(global_adj)
    for u in range(n):
        global_adj[u] = [v for v in global_adj[u] if u != v]
        
    for u in range(n):
        adj = copy.deepcopy(global_adj)
        path = [u]
        curr = u
        while adj[curr]:
            next = max(adj[curr], key = lambda v: len(adj[v]))
            for v in range(n):
                adj[v] = [w for w in adj[v] if w != curr]
            path.append(next)
            curr = next
        results.append(path)

    return results

print(len(graphs))
for adj in graphs:
    approx_paths = find_greedy_paths(adj)
    best_path = max(approx_paths, key = lambda p: len(p))
    print(len(best_path))
    print(*best_path)

