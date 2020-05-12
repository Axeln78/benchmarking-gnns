import numpy as np 
import dgl

def remove_random_edges(G,n):
    ids = np.random.randint(0,G.number_of_edges(), size = n)
    G.remove_edges(ids)
    
def rm_rand_edges(bG,rate):
    '''
    rate: as a %
    '''
    g_arr = dgl.unbatch(bG)
    for g in g_arr:
        g = remove_random_edges(g,int(g.number_of_edges()*rate))
    return dgl.batch(g_arr)

