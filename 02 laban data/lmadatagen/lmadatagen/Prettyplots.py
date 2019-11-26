#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# In[2]:


joint_hier = [
    ('head', 'neck', 'blue'),
    ('neck', 'root', 'darkred'),
    ('root', 'clavicle', 'brown'),
    ('neck', 'leftShoulder', 'red'), 
    ('leftShoulder', 'leftElbow', 'darkred'), 
    ('leftElbow', 'leftWrist', 'orange'),
    ('neck', 'rightShoulder', 'orange'), 
    ('rightShoulder', 'rightElbow', 'lightgreen'), 
    ('rightElbow', 'rightWrist', 'green'),
    ('clavicle', 'leftHip', 'green'), 
    ('leftHip', 'leftKnee', 'lightgreen'), 
    ('leftKnee', 'leftAnkle', 'lightblue'),
    ('clavicle', 'rightHip', 'lightblue'), 
    ('rightHip', 'rightKnee', 'cyan'), 
    ('rightKnee', 'rightAnkle', 'blue')
]


# In[3]:


def plt_keyframe_curve(ax, keyframes, curve):
    curve = curve / np.amax(curve)
    ax.plot(range(0, len(curve)), curve, 'r', alpha=0.9)
    ax.scatter(keyframes, curve[keyframes], color='k', marker="$F$")
    ax.set_xlim(0, len(curve))
    ax.set_ylim(0.0, 1.2)
    ax.set(xlabel='Samples [fs=30]')
    ax.set(ylabel=r'Normalized angle $\theta$, $\theta$ $\epsilon$ [0, $\pi$]')
    ax.legend(['Curve', 'Keyframe'])


# In[4]:


def plt_keyframe_skeleton(ax, keyframes, positions, scale=200.):
    pos = positions
    pos['root'] = np.zeros_like(pos['head'])
    pos['neck'] = (pos['leftShoulder'] + pos['rightShoulder']) / 2.
    pos['clavicle'] = (pos['leftHip'] + pos['rightHip']) / 2.
    
    for frame in keyframes:
        lines = []
        for f, t, c in joint_hier:
            p1 = pos[f][frame]
            p2 = pos[t][frame]
            x = [p1[0]*scale + frame, p2[0]*scale + frame]
            y = [p1[1], p2[1]]
            lines.append(Line2D(x, y, color=c))
            ax.scatter(x, y, color=c, alpha=0.9)
            
        for l in lines:
            ax.add_line(l)
        
    ax.set_xlim(min(keyframes) - scale//10, max(keyframes) + scale//10)
    ax.set_ylim(scale, -scale)
    #ax.set_ylim(0.5*scale, -0.25*scale)
    ax.set_xticks(tuple(keyframes))
    ax.set_xticklabels(keyframes)
    ax.set(xlabel='Keyframes')
    


# In[ ]:




