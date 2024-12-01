import graphviz
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self):
        self.adj_map = {}
    
    def add_vertex(self, vertex):
        if vertex not in self.adj_map:
            self.adj_map[vertex] = []
    
    def add_edge(self, vertex1, vertex2):
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.adj_map[vertex1].append(vertex2)
    
    def save_to_png(self):
        dot = graphviz.Digraph()
        for vertex in self.adj_map:
            dot.node(str(vertex))
            for neighbor in self.adj_map[vertex]:
                dot.edge(str(vertex), str(neighbor))
        dot.render('graph', format='png', cleanup=True)

    @staticmethod
    def complete_graph(n):
        g = Graph()
        for i in range(n):
            for j in range(i+1, n):
                g.add_edge(i, j)
                g.add_edge(j, i)
        return g
    
    @staticmethod
    def random_graph(n, p):
        g = Graph()
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < p:
                    g.add_edge(i, j)
                    g.add_edge(j, i)
        return g