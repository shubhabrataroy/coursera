# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:42:01 2014
You should have received an archive together with this document
(MD5=9eed1dc629e3e99f248be11508ba7d07). The archive contains contains two files.
The first file, {\em hk_dataset_for_independent_work_signals.csv}, is a list of individual signals, where
each row contains the user identifier, the base station identifier (cell ID), the time in seconds
since epoch at UTC (the local time of the data is the Hong Kong timezone), and the event type.
All signals from one operator that occurred in three selected LACs are recorded. There are
three LAC: 301, 501, and 410. The second file, {\em hk_dataset_for_independent_work_lac.txt},
contains the grouping of cell IDs by LAC for all cell IDs present in the signals dataset file.
These three LACs contain two shopping malls in Hong Kong:
\begin{itemize}
\item Mall T, with CIDs: 5330, 5340, 5350, 5360, 5370, 5380, 5390, 5400, 5410, 131070,
131080, 131090, 131100, 131110, 131120, 192030, 192040, 192050, 192060, 103750,
103760, 103770 and 103780.
\item Mall C: 107340, 107370, 107400, 107350, 107380, 107410, 122910, 122940, 122970,
122960, 122990, 122930, 122920, 122980 and 122950.
Cell IDs that are not in the lists above do not belong to those malls, but to their surroundings, as
defined by the LAC groupings.
\end{itemize}
Although all identifiers have been scrambled, the data remain consistent.
The events in this dataset do not actually correspond to those described in the short introduction
above. This introduction was more centered on GSM network, while the signals present in this
dataset are exclusively GPRS network signals.
\begin{itemize}
\item Event type 1 is a PDP Context creation event,
\item event type 2 is a PDP update event
\end{itemize}
Both are events of a control protocol of GPRS, so the corresponding signals do not carry data.
From your point of view, those packets are most useful for their location value, not so much for
the event type, as making use of that would require you to delve into the details of GPRS.
@author: sroy
"""
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from functools import partial
from os import listdir, getcwd, makedirs
from os.path import isfile, join, abspath, exists, isdir
import sys
from shutil import copy
import getopt
import pylab
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import *

path = '/home/sroy/Desktop/Personal/Teralytics/hk_2_malls_dataset'
mobiledata= pd.read_csv(join(path,'hk_dataset_for_independent_work_signals.csv'))

def remove_consecutive_duplicates(a):
    last = None
    for x in a:
        if x != last:
            yield x
        last = x
def return_most_frequent_list(X):
    max_count = 0
    for e in X:
        if (X.count( e )> max_count):
            most_frequent_list  = e
            max_count = X.count( e )
    return (most_frequent_list,max_count)

# check for duplicate data
#A1 = len(mobiledata['uid'])
#A2 = len(np.unique(mobiledata['uid']))
#A3 = len(np.unique(mobiledata['cid']))

userid = mobiledata['uid']
cellid = mobiledata['cid']
time = mobiledata['time']
time = time - time[0]
eventtype = mobiledata['event_type']

CID_T = [5330, 5340, 5350, 5360, 5370, 5380, 5390, 5400, 5410, 131070, 131080, 131090, 131100, 131110, 131120, 192030, 192040, 192050, 192060, 103750, 103760, 103770, 103780]
CID_C = [107340, 107370, 107400, 107350, 107380, 107410, 122910, 122940, 122970, 122960, 122990, 122930, 122920, 122980, 122950]


CID_410 = [31490,116630,105100,107790,104470,17310,117360,31010,5340,107820,126260,109630,195980,121680,5330,104410,2140,104420,105190,108520,87280,105790,30840]
CID_301 = [110080,202240,196610,125960,202250,113680,125970,202260,113690,125980,202330,202270,113700,125990,122920,126130,122930,195130,122940,110090,124480,125280,122950,124490,122960,124500,202340,122970,109660,124510,172560,122980,124520,126140,113770,202350,124530,113780,202360,107130,124540,107370,113790,202370,107140,124550,110730,202380,107150,126100,122990,193690,108700,110750,197280,126150,108710,126120,197290,108720,72370,122910,107380,108730,195260,108740,195270,108750,126160,200410,194780,109690,196830,72400,200420,110310,194280,110320,202280,194290,110330,194300,202010,202020,108850,108860,126090,72340,114500,108870,107340,114510,121680,107350,114520,121690,107360,121700,77670,125290,115570,125300,175990,110740,115580,107390,201600,77700,115590,107400,201610,107410,176020,107420,201630,77730,201640,194990,201650,126110,195000,202230,116690,116700,194810,116710,125930,172530,125940,110070,196600,201620,125950]
CID_501 = [104960,104450,123400,104970,104460,131090,104470,105220,131100,119130,192030,104480,195910,131110,192040,104490,131120,131080,192050,104500,192060,104510,192580,104520,192590,104530,105230,104540,104550,123410,192640,199190,192650,105100,114320,5400,105110,114330,105120,194210,114340,194220,192700,196800,199200,192710,192290,5330,5340,202470,202450,5360,105210,199210,5380,196360,192300,5390,105240,201050,194010,5410,119080,202460,107820,119090,126260,192310,199220,195900,105790,192320,107830,119110,126280,192330,119120,126290,192340,108430,104420,103770,126300,192350,103780,5350,126310,192360,119100,103790,113520,192370,126270,103800,113530,192380,107840,113540,126350,113040,108440,126360,113050,105200,108450,113060,103750,193010,105800,112050,112060,108490,126370,105190,108500,104410,5370,108510,103760,194020,108520,105810,104430,108530,123390,104950,104440,108540,131070]

# Check intersection between LAC and malls
whichLAC_C = len(CID_C) - len(set(CID_301).intersection(CID_C)) # Zero means mall C falls in LAC 301
whichLAC_T = len(CID_T) - len(set(CID_501).intersection(CID_T)) # Zero means mall T falls in LAC 501

""" Daily mall statistics """

#bins1 = np.histogram(time)[1]

whichmall = []
for i in range(0, len(cellid)):
    if cellid[i] in CID_T:
        whichmall.append(0)
    elif cellid[i] in CID_C:
        whichmall.append(1)
    else:
        whichmall.append(2)
        
mobiledata['whichmall'] = whichmall
bins = np.arange(time[0], time[len(time)-1], 3600) # creates a bin of 24 hours
# Statistics for mall T
mobiledata_T = mobiledata[mobiledata['whichmall'] == 0]
userid_T = mobiledata_T['uid']
cellid_T = mobiledata_T['cid']
time_T = mobiledata_T['time']
time_T = time_T - time_T[1]
eventtype_T = mobiledata_T['event_type']


mobiledata_T['pred_bin'] = np.digitize(time_T, bins)

aggregated_uid_T = mobiledata_T.groupby('pred_bin').count()['uid']


# Statistics for mall C

# Statistics for mall T
mobiledata_C = mobiledata[mobiledata['whichmall'] == 1]
userid_C = mobiledata_C['uid']
cellid_C = mobiledata_C['cid']
time_C = mobiledata_C['time']
time_C = time_C - time_C[17]
eventtype_C = mobiledata_C['event_type']


mobiledata_C['pred_bin'] = np.digitize(time_C, bins)

aggregated_uid_C = mobiledata_C.groupby('pred_bin').count()['uid']

# Statistics for territories of the malls
mobiledata_O = mobiledata[mobiledata['whichmall'] == 2]
userid_O = mobiledata_O['uid']
cellid_O = mobiledata_O['cid']
time_O = mobiledata_O['time']
time_O = time_O - time_O[0]
eventtype_O = mobiledata_O['event_type']


mobiledata_O['pred_bin'] = np.digitize(time_O, bins)

aggregated_uid_O = mobiledata_O.groupby('pred_bin').count()['uid']

# compare the correlations:
sp.corrcoef(aggregated_uid_C,aggregated_uid_T)
sp.corrcoef(aggregated_uid_C,aggregated_uid_O)
sp.corrcoef(aggregated_uid_T,aggregated_uid_O)

## find the path of an user (uid) in terms of cell towers (cid) along time

unique_userid = list(set(userid))
average_time = []
n_hop = []
for j in unique_userid:
    X = mobiledata[mobiledata['uid'] == j]
    time_diff = X['time'].diff()
    idx = time_diff.index[0]
    time_diff[idx] = 0
    X['time_diff'] = time_diff
    unique_cid = list(set(X['cid']))
   # list_cellid = list(remove_consecutive_duplicates(cellid))
    n_hop.append(len(unique_cid)) # average number of cids covered by an user
    for k in unique_cid:
        Y = X[X['cid'] == k]
        total_time = Y['time_diff'].sum()
    average_time.append(mean(total_time))
average_time = [x/3600 for x in average_time]    
# how many values for less than 30 mins or 1hr average time
sum(i < 0.5 for i in average_time)
# how many people travel less than three hops
sum(i < 2 for i in n_hop)

# find the most popular path (first )

pathlist = []
for j in unique_userid:
    X = mobiledata[mobiledata['uid'] == j]
    cellid = X['cid']
    list_cellid = list(remove_consecutive_duplicates(cellid))
    pathlist.append(list_cellid)
    
path_mod = [i for i in pathlist if len(i) > 3]
    
    
L = return_most_frequent_list(path_mod)

## clustering male vs female
# computing K-Means with K = 2 (2 clusters)
x = np.vstack(average_time)
y = np.vstack(n_hop)
data = np.concatenate((x,y), axis = 1)

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,3)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
#plot(data[idx==0,0],data[idx==0,1],'ob',
#     data[idx==1,0],data[idx==1,1],'or')
#plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
#show()

plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or',
     data[idx==2,0],data[idx==2,1],'og') # third cluster points
plot(centroids[:,0],centroids[:,1],'sm',markersize=8)
show()

idx1 = [i for i,val in enumerate(idx) if val==0]
idx2 = [i for i,val in enumerate(idx) if val==1]
idx3 = [i for i,val in enumerate(idx) if val==2]

average_time1 = []
n_hop1 = []
for i in idx1:
    average_time1.append(average_time[i])
    n_hop1.append(n_hop[i])
mean(average_time1)
mean(n_hop1)    

    
    
        
    
    
    
    
