import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from profile import PreferenceProfile
from distinctipy import get_colors
from ballot import Ballot
from typing import Optional, Callable
from pydantic import BaseModel, validator


def build_graph(n):

    Gc = nx.Graph()

        # base cases
    if n==1:
        Gc.add_nodes_from([(1)])
    elif n==2:
        Gc.add_nodes_from([(1,2),(2,1)])
        Gc.add_edges_from([((1,2), (2,1))])

    else:
            
        if n-1 not in Graphs.keys(): 
            # make the adjacency graph of size (n - 1)
            build_graph(n-1)
        G_prev = Graphs[n-1]
        for i in range(1,n+1):
                # add the node for the bullet vote i
            Gc.add_node(tuple([i]))

            # make the subgraph for the ballots where i is ranked first
            G_corner = relabel(G_prev,i,n)
      
                # add the components from that graph to the larger graph
            Gc.add_nodes_from(G_corner.nodes)
            Gc.add_edges_from(G_corner.edges)

                # connect the bullet vote node to the appropriate verticies
            if n == 3:
                Gc.add_edges_from([(k,tuple([i])) for k in G_corner.nodes])
            else:
                Gc.add_edges_from([(k,tuple([i])) for k in G_corner.nodes if len(k) == 2])
            
        nodes = Gc.nodes

            # add the additional edges corresponding to swapping the order of the
            # first two candidates
        new_edges = []
        for k in nodes:
            if len(k)==2:
                new_edges.append(((k[0],k[1]),(k[1],k[0])))
            elif len(k)>2:
                l = list(k)
                a = l[0]
                b=l[1]
                new_edges.append((tuple([a]+[b]+l[2:]), tuple([b]+[a]+l[2:])))

        Gc.add_edges_from(new_edges)

    Graphs[n] = Gc
    return Gc


def relabel(gr, new_label, num_cands):
    node_map = {}
    graph_nodes = list(gr.nodes)


    for k in graph_nodes:
            # add the value of new_label to every entry in every ballot
        tmp = [new_label+y for y in k]

            # reduce everything mod new_label
        for i in range(len(tmp)):
            if tmp[i]> num_cands:
                tmp[i] = tmp[i]- num_cands
        node_map[k] = tuple([new_label]+tmp)

    return nx.relabel_nodes(gr, node_map)


Graphs = {}


class BallotGraph():
    profile: PreferenceProfile
    ballot_dict: dict
    num_cands: int
    num_voters: int
    
    
    def __init__(self,profile):
        self.profile = profile
        self.ballot_dict= profile.to_dict()
        
        #cand_list = []
        #for key in ballot_dict.keys():
        #    for i in key:
        #        cand_list.append(i)
      
        self.num_cands = len(self.profile.get_candidates())
        
        if self.num_cands not in Graphs.keys():
            build_graph(self.num_cands)
        
        
        all_ballots = Graphs[self.num_cands].nodes
        di = {}
        for ballot in all_ballots:
            di[ballot] = 0
        
        self.ballot_dict = di | ballot_dict
        
        
        self.clean()
        
        self.num_voters = sum(ballot_dict.values())
        
        
    def clean(self): #deletes empty ballots, changes n-1 length ballots to n length ballots and updates counts
        di = self.ballot_dict.copy()
        
        for ballot in di.keys():
            if len(ballot)==0:
                self.ballot_dict.pop(ballot)
            elif len(ballot)==self.num_cands-1:
                for i in self.profile.get_candidates():
                    if i not in ballot:
                        self.ballot_dict[ballot+tuple([i])]+= di[ballot]    
                        self.ballot_dict.pop(ballot)
                        break
                
        
    def visualize(self, neighborhoods = {}):
        Gc = Graphs[self.num_cands]    
        if neighborhoods == {}:
            self.clean()
            WHITE = (1,1,1)
            BLACK=(0,0,0)
            cols = get_colors(self.num_cands, [WHITE,BLACK])
            node_cols = []
            ballots = list(Gc.nodes)
        
            for bal in Gc.nodes:
                if self.ballot_dict[bal]!=0:
                    i = self.profile.get_candidates().index(bal[0])
                    node_cols.append(cols[i])
                else:
                    node_cols.append(WHITE)
            ##want to include number of votes as part of labels,  color ballots with 0 votes grey
        
            nx.draw_networkx(Gc,with_labels = True, node_color = node_cols)
            
        else:
            WHITE = (1,1,1)
            BLACK = (0,0,0)
            cols = get_colors(len(neighborhoods), [WHITE, BLACK])
            node_cols = []
            centers = list(neighborhoods.keys())
            
            for bal in Gc.nodes:
                found = False
                for i in range(len(centers)):    
                    weight = (neighborhoods[centers[i]])[1]
                    if bal in (neighborhoods[centers[i]])[0].nodes:
                        node_cols.append(tuple([(1-self.ballot_dict[bal]/weight)*x for x in cols[i]]))
                        found = True
                        break ##breaks the inner for loop
                if not found:
                    node_cols.append(WHITE)
            nx.draw_networkx(Gc, with_labels = True, node_color = node_cols)
        return
            
        
    def distance_between_subsets(self, A,B):
        return min([nx.shortest_path_length(Graphs[self.num_cands], a, b) for a in A.nodes for b in B.nodes])
        
    @staticmethod
    def show_all_ballot_types(n):
        if n not in Graphs.keys():
            build_graph(n)
        Gc = Graphs[n]
        nx.draw(Gc, with_labels = True)
        plt.show()
  
    def compare(self, new_pref: profile, dist_type: Callable):
        return  ##to be completed
    
    def compare_rcv_results(self, new_pref):
        return ##to be completed
    
    def subgraph_neighborhood(self,center,radius = 2):
        return nx.ego_graph(Graphs[self.num_cands],center,radius)
    
    def k_heaviest_neighborhoods(self, k=2, radius=2):
        cast_ballots = set([x for x in self.ballot_dict.keys() if self.ballot_dict[x] > 0]) ##has 
            
        max_balls = {}
            
        for i in range(k):
            weight = 0
            if len(cast_ballots)==0:
                break
            for center in cast_ballots:
                tmp = 0
                ball = self.subgraph_neighborhood(center, radius)
                relevant = cast_ballots.intersection(set(ball.nodes))##cast ballots inside the ball
                for node in relevant: 
                    tmp += self.ballot_dict[node]

                if tmp>weight:
                    weight = tmp
                    max_center = center
                    max_ball = ball 
                
            not_cast_in_max_ball = set(max_ball.nodes).difference(cast_ballots)
            max_ball.remove_nodes_from(not_cast_in_max_ball)
            max_balls[max_center] = (max_ball, weight)
                
            cast_ballots =  cast_ballots.difference(set(max_ball.nodes))
                
        return max_balls

   

