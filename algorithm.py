import torch
import copy
from py_extraction.feature_extraction import *
import heapq


def beam_search_path(adj, model, width=3):
    """
    Beam search for the longest/best path in the graph.

    Each beam maintains its own:
      - score: expected length of full path
      - current vertex: last vertex in the current path
      - last beam: previous beam that resulted in current beam
      - graph state: vertices removed as the path grows

    At every step each beam extends by its locally best candidate;
    then only the top `width` beams (by score) survive.

    Returns a list of paths
    """

    def clone(x): 
        return copy.deepcopy(x)
    
    def remove_vertex(adj, u):
        n = len(adj)
        return [[w for w in adj[v] if w != u] if v != u else [] for v in range(n)]

    model.eval()
    device = next(model.parameters()).device
    n = len(adj)

    
    # Score every node on the original graph
    feats = torch.as_tensor(extract_features(adj), dtype=torch.float32, device=device)

    with torch.no_grad():
        initial_beams = []
        for u in range(n):
            heapq.heappush(initial_beams, (model(feats[u:u + 1]).squeeze().item(), u, -1, clone(adj)))
            if len(initial_beams) > width:
                heapq.heappop(initial_beams)


    # Start beam search as long as there are remaining beams
    beam_stages = [initial_beams]
    while True:
        beams = beam_stages[-1]
        num_beams = len(beams)

        beams_candidates = []

        # Find all possible branches
        for beam_index in range(num_beams):
            _, curr, _, adj = beams[beam_index]

            neighbors = adj[curr]
            adj = remove_vertex(adj, curr)
            feats = torch.as_tensor(extract_features(adj), dtype=torch.float32, device=device)
            with torch.no_grad():
                beams_candidates.extend((model(feats[v: v + 1]).squeeze().item(), v, beam_index, clone(adj)) for v in neighbors)

        if not beams_candidates:
            break

        # Take top width next branches
        next_beams = []
        for beam in beams_candidates:
            heapq.heappush(next_beams, beam)
            if len(next_beams) > width:
                heapq.heappop(next_beams) 

        beam_stages.append(next_beams)

    path = []
    beam_index = 0
    for beam_stage in range(len(beam_stages) - 1, -1, -1):
        path.append(beam_stages[beam_stage][beam_index][1])
        beam_index = beam_stages[beam_stage][beam_index][2]
    
    return path[::-1]