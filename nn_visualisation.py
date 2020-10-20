import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Slider


class Visualization:

    colors = ['r', 'g', 'b', 'y']

    def __init__(self, model):
        self.model = model
        self.graph_size_x=1000
        self.graph_size_y=1000
        plt.ion()

    def add_drawing(self,m1,learning_error,train_data,test_data):
        plt.clf()
        G = nx.Graph()
        k=0
        edges=[]
        edge_values=[]
        n_actvalues=[]
        k0=0 # first neuron number in the layer
        for i in range(self.model.num_of_layers):
            k0=k
            for j in range(self.model.layers[i]):

                G.add_node(k, pos=((i+1)*self.graph_size_x/(self.model.num_of_layers+1), (j+1)*self.graph_size_y/(self.model.layers[i]+1)))
                n_actvalues.append(self.model.actvalues[i][j])
                # we add edges to all previous layer vortexes
                if i>0:
                    for p_j in range(self.model.layers[i-1]):
                        edges.append([k0-self.model.layers[i-1]+p_j,k])
                        edge_values.append(self.model.weights[i-1][j,p_j])
                k += 1

        plt.subplot(221)
        pos = nx.get_node_attributes(G, 'pos')
        norm_n=matplotlib.colors.Normalize(vmin=0.2,vmax=.8)
        cmap_n=plt.get_cmap('rainbow')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=range(k),
                               node_color=n_actvalues,
                               node_cmap=cmap_n,
                               node_size=10,
                               alpha=0.8)
        sm_n = plt.cm.ScalarMappable(cmap=cmap_n, norm=norm_n)
        sm_n.set_array([])
        plt.colorbar(sm_n)
        norm = matplotlib.colors.Normalize(vmin=min(edge_values), vmax=max(edge_values))
        cmap=plt.get_cmap('viridis')
        nx.draw_networkx_edges(G, pos,
                               edgelist=edges,
                               width=2, alpha=0.8, edge_color=edge_values,
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
        plt.subplot(223)
        plt.plot(learning_error)
        plt.title("Learning process")
        plt.subplot(224)
        input_size = m1.layers[0] - int(m1.with_bias)
        output_size = m1.layers[-1]
        di = np.zeros(input_size)
        if not(test_data is None):
            results = np.zeros(len(test_data))
            for i in range(len(test_data)):


                di[0] = test_data[i][0]
                results[i] = m1.evaluate(di)
            plt.scatter(test_data.T[0], results, marker='.', label='Regression')
            plt.scatter(test_data.T[0], test_data.T[1], marker='.', label='Test data')
        else:
            results = np.zeros(len(train_data))
            for i in range(len(train_data)):


                di[0] = train_data[i][0]
                results[i] = m1.evaluate(di)
            plt.scatter(train_data.T[0], results, marker='.', label='Regression')



        plt.scatter(train_data.T[0], train_data.T[1], marker='.', label='Train data')
        plt.legend()
        plt.suptitle('NN graph')
        """plt.subplots_adjust(bottom=0.1)
        fig,ax=plt.subplots()
        axinp=plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='red')
        s_inp=Slider(axinp,"Input",self.model.input_range[0],self.model.input_range[1],valinit=self.model.input_range[0])"""
        plt.draw()  # display
        plt.pause(0.0001)