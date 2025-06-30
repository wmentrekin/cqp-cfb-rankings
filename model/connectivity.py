
import networkx as nx

def compute_connectivity_index(games, teams):
    '''
    Compute the connectivity index C / |T| where:
    - games is a list of tuples (team1, team2)
    - teams is a list of all FBS team names
    '''
    G = nx.Graph()
    G.add_nodes_from(teams)
    for i, j in games:
        G.add_edge(i, j)
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / len(teams)
