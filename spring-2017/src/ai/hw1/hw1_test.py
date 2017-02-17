import networkx as nx

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

q = Q.PriorityQueue()

q = Q.PriorityQueue()
q.put((10,'ten'))
q.put((1,'one'))
q.put((5,'five'))