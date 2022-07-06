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


# Haversine Formula를 이용해 인접 정거장 사이 거리를 계산. 이 거리를 Weight로 하여 버스 노선을 Weighted network를 구현.

# In[2]:


bus_raw=pd.read_csv('C:/Users/INHA/Desktop/BP/20220420.csv', encoding='cp949')  
bus_raw.columns = ['line_ID','line','num','NODE_ID','ARS-ID', 'station', 'lat','lon']
bus_raw.head()


# 모든 노선을 다 하면 메모리 초과. 랜덤 샘플을 이용해서 추출

# In[5]:


ids=pd.unique(bus_raw['line_ID'])
selec=np.random.choice(ids, 10, replace=False)


# In[7]:


bus_raw=bus_raw[bus_raw.apply(lambda row: row['line_ID'] in selec, axis=1)]


# In[6]:


ids=pd.unique(bus_raw['line_ID'])


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


# 정거장의 가능한 모든 pair를 데이터 프레임으로 만든 후 각각 직선거리(dist=distance)와 버스 노선을 이용한 최단 이동 거리(spl=shortest path length)를 계산.

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


# spl과 dist의 비율 ratio를 계산. ratio가 클수록 비효율적임.

# In[725]:


comb['ratio']=comb['spl']/comb['dist']


# In[727]:


comb.to_csv('bus_comb_with_ratio.csv')
comb.head()

