import os
import numpy as np
import scipy as scp
#import pylab
import math
import matplotlib.pyplot as plt
import networkx as nx 
r = 0.05   # The interest rate of the riskless bond
p = 0.5   # The probability for the price at each period to go upwarwd
q = 1.0-p # The probability for the price at each period to go downwarwd
u = 2.0   # The level of increase of the price
d = 0.5   # The level of decrease of the price
T = 5     # Number of periods in the model
S0 = 100  # Value of the risky asset at time 0

if (d < 1+r) & (u>1+r) :
    print("Marché viable")
else :
    print("Arbitrage")

p_star = (1.0+r-d)/(u-d)
print("p*=",p_star)  

M = 10    # The number of simulated sample paths
S = np.zeros((M,T+1))

start = np.zeros(M)
for i in range(0,M):
    start[i]=S0
S[:,0]=start

B = np.random.binomial(1,p,T+1) # We simulate all the Bernouilli random variables

def transform(B): # Transform a vector of O and 1 into a vector of u and d's
    temp1 = np.size(B)
    up_and_down = np.zeros(temp1)
    for i in range(0,temp1):
        if B[i]==1:
            up_and_down[i]=u
        else:
            up_and_down[i]=d
    return up_and_down

def binomial_grid(n,s0):
    G=nx.Graph() 
    for i in range(0,n):
        j=-i+1
        while (j<i+2):
            G.add_edge((i,j),(i+1,j+1),weight=0.0)
            G.add_edge((i,j),(i+1,j-1),weight=0.0)
            j=j+2
    
    j=1
    for i in range(0,n):
        r=np.random.binomial(1,p,1)
        if r >0:
            G.add_edge((i,j),(i+1,j+1),weight=1.0)
            j=j+1
        else:
            G.add_edge((i,j),(i+1,j-1),weight=1.0)
            j=j-1
    
    posG={}
    lab={}
    for node in G.nodes():
        posG[node]=(node[0],node[1])
        if node[0]==0:
            lab[node]=s0
        i=node[0]
        j=node[1]
        if j>1:
            lab[node]=s0*(u**(j-1))
        elif j==1:
            lab[node]=s0
        else:
            lab[node]=s0*(d**(1-j))
    elarge=[(a,b) for (a,b,c) in G.edges(data=True) if c['weight'] >0.5]
    esmall=[(a,b) for (a,b,c) in G.edges(data=True) if c['weight'] <=0.5]
    nx.draw_networkx_edges(G,posG,edgelist=elarge,edge_color='blue',width=2)
    nx.draw_networkx_edges(G,posG,edgelist=esmall,style='dashed')
    nx.draw_networkx_labels(G,posG,lab,font_size=15,font_family='sans-serif')
    plt.show()

    binomial_grid(5,100.0)

    def plot_tree(g):
        pos={}
        lab={}
        
        for n in g.nodes():
            pos[n]=(n[0],n[1])
            lab[n]=float(int(g.nodes[n]['value']*1000))/1000 # This is just to print only with 10^-2 precision
        
        elarge=g.edges(data=True)
        nx.draw_networkx_edges(g,pos,edgelist=elarge)
        nx.draw_networkx_labels(g,pos,lab,font_size=15,font_family='sans-serif')
        plt.show() 

    def graph_stock():
        S=nx.Graph()
        for i in range(0,T):
            j=-i+1
            while (j<i+2):
                S.add_edge((i,j),(i+1,j+1))
                S.add_edge((i,j),(i+1,j-1))
                j=j+2
                
        for n in S.nodes():
            
            if n[0]==0:
                S.nodes[n]['value']=S0
            i=n[0]
            j=n[1]
            if j>1:
                S.nodes[n]['value'] = S0*(u**(j-1))
            elif j==1:
                S.nodes[n]['value'] = S0
            else:
                S.nodes[n]['value'] = S0*(d**(1-j))
        return S
    
    S=nx.Graph()
    S =graph_stock()
    plot_tree(S)