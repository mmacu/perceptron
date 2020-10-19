import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Visualization:

    colors = ['r', 'g', 'b', 'y']

    def __init__(self, model):
        self.model = model
        self.graph_size_x=1000
        self.graph_size_y=1000

    def draw_solution(self):
        G = nx.Graph()
        k=0
        edges=[]
        edge_values=[]
        k0=0 # first neuron number in the layer
        for i in range(self.model.num_of_layers):
            k0=k
            for j in range(self.model.layers[i]):

                G.add_node(k, pos=((i+1)*self.graph_size_x/(self.model.num_of_layers+1), (j+1)*self.graph_size_y/(self.model.layers[i]+1)))

                # we add edges to all previous layer vortexes
                if i>0:
                    for p_j in range(self.model.layers[i-1]):
                        edges.append([k0-self.model.layers[i-1]+p_j,k])
                        edge_values.append(self.model.weights[i-1][j,p_j])
                k += 1


        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=range(k),
                               node_color='b',
                               node_size=50,
                               alpha=0.8)
        norm = matplotlib.colors.Normalize(vmin=min(edge_values), vmax=max(edge_values))
        cmap=plt.get_cmap('viridis')
        nx.draw_networkx_edges(G, pos,
                               edgelist=edges,
                               width=5, alpha=0.8, edge_color=edge_values,
                               edge_cmap=cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)


        """       i = 0
        for layer in self.model.layers:
            previous_vertex = 0
            edges = []
            for vertex in route:
                edges.append((previous_vertex, vertex))
                previous_vertex = vertex
            edges.append((previous_vertex, 0))
            G.add_edges_from(edges)
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edges,
                                   width=4, alpha=0.5, edge_color=self.colors[i % len(self.colors)])
            i += 1
        nx.draw(G, pos, with_labels=True, node_size=2, font_size=10, width=1.2)"""

        plt.suptitle('NN graph')
        plt.show()  # display
