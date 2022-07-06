#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx
from haversine import haversine
from itertools import permutations


# In[2]:


bus_raw=pd.read_csv('C:/Users/INHA/Desktop/BP/20220420.csv', encoding='cp949')  
bus_raw.columns = ['line_ID','line','num','NODE_ID','ARS-ID', 'station', 'lat','lon']
bus_raw.head()


# In[3]:


ids=pd.unique(bus_raw['line_ID'])
ids.sort()


# In[4]:


ids


# In[5]:


ids=pd.unique(bus_raw['line_ID'])
ids


# In[6]:


ids=pd.unique(bus_raw['line_ID'])
len(ids)
selec=np.random.choice(ids, 3, replace=False)
selec=ids[1:10]


# In[7]:


bus_raw=bus_raw[bus_raw.apply(lambda row: row['line_ID'] in selec, axis=1)]


# In[8]:


ids=pd.unique(bus_raw['line_ID'])
len(ids)


# In[9]:


ids=pd.unique(bus_raw['NODE_ID'])
lines=pd.unique(bus_raw['line_ID'])
G = nx.Graph()
G.add_nodes_from(ids)


# In[715]:


for i in lines:
    line=bus_raw[bus_raw['line_ID']==i]
    line['des']=line['NODE_ID'].shift(-1)
    line['X']=line['lat'].shift(-1)
    line['Y']=line['lon'].shift(-1)
    line=line.dropna()
    line['des']=line['des'].astype('int')
    line['distance'] = line.apply(lambda row: haversine((row['X'], row['Y']), (row['lat'], row['lon'])), axis=1)
    G.add_weighted_edges_from(line[['NODE_ID','des','distance']].to_numpy())


# In[716]:


plt.figure(figsize=(50,30))
nx.draw(G, with_labels=False, font_weight='bold')
plt.savefig('bus_graph.png')


# In[717]:


comb=pd.DataFrame(list(permutations(ids, 2)), columns=['ori','des'])


# In[718]:


comb


# In[719]:


def dist(ori, des):
    ori_corr=bus_raw[bus_raw['NODE_ID']==ori][['lat','lon']]
    des_corr=bus_raw[bus_raw['NODE_ID']==des][['lat','lon']]
    return  haversine(np.array(ori_corr)[0], np.array(des_corr)[0],unit = 'km')


# In[720]:


def spl(ori,des):
    return nx.dijkstra_path_length(G,ori,des)


# In[721]:


comb['spl']=comb.apply(lambda row: spl(row['ori'],row['des']), axis=1)


# In[722]:


comb['dist']=comb.apply(lambda row: dist(row['ori'],row['des']), axis=1)


# In[723]:


comb.to_csv('bus_comb.csv')


# In[724]:


comb.head()


# In[725]:


comb['ratio']=comb['spl']/comb['dist']


# In[727]:


comb.to_csv('bus_comb_with_ratio.csv')
comb.head()


# In[730]:


comb_sort=comb.sort_values('ratio')


# In[127]:


line[['NODE_ID','des','distance']].to_numpy()


# In[780]:


comm_long=comb_sort[comb_sort['dist']>5]
comm_selet=comm_long[comm_long['ratio']>4]
comm_selet.reset_index()
comm_selet.to_csv('final.csv')


# In[768]:


plt.scatter(bus_raw['lat'],bus_raw['lon'],s=5)


# In[ ]:


def inter_node(ori, des,c):
    ori_lat=bus_raw[bus_raw['NODE_ID']==ori] ['lat']
    ori_lon=bus_raw[bus_raw['NODE_ID']==ori] ['lon']
    des_lat=bus_raw[bus_raw['NODE_ID']==des] ['lat']
    des_lon=bus_raw[bus_raw['NODE_ID']==des] ['lon']
    
    x=np.ravel([ori_lat,des_lat])
    y=np.ravel([ori_lon,des_lon]) 

    plt.plot(x,y,c)


# In[779]:


plt.scatter(bus_raw['lat'],bus_raw['lon'],s=5)
inter_node(108000016,110000145,'r')
plt.savefig('example.png')

