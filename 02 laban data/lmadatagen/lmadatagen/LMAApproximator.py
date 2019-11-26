#!/usr/bin/env python
# coding: utf-8

import random
import pandas as pd
import matplotlib.pyplot as plt
import lmadatagen.Prettyplots as pp
import numpy as np
from functools import reduce
from sklearn.cluster import KMeans
from scipy.signal import lfilter, gaussian, convolve
from lmadatagen import print_progress

def np_step(x, a):
    boolean_arr = np.zeros_like(x).astype(np.int32)
    for col, predicate in enumerate(a):
        boolean_arr[x[:, col] >= predicate, col] = 1
        
    return boolean_arr

def curvature(curve_vel, curve_acc):
    dx_dt, dy_dt = curve_vel[:, 0], curve_vel[:, 1]
    d2x_dt2, d2y_dt2 = curve_acc[:, 0], curve_acc[:, 1]
    return np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    
def length(vector, axis=1):
    return np.linalg.norm(vector, ord=None, axis=axis)

def angle_btw(v1, v2):
    return np.array([ np.arccos(np.dot(v1[r], v2[r]) / (length(v1[r], 0)*length(v2[r], 0))) for r in range(0, v1.shape[0]) ])

def lincomb(v_array, weights):
    wsum = np.zeros((v_array.shape[0], ))
    for col in range(0, v_array.shape[1]):
        wsum += weights[col]*v_array[:, col]
        
    return wsum

def curve_salience(curve, wlen=5, sig=1):
    kern_lo = gaussian(wlen, std=sig)
    kern_hi = gaussian(wlen, std=2.0*sig)
    y_lo = convolve(curve, kern_lo, mode='same')
    y_hi = convolve(curve, kern_hi, mode='same')
    return np.abs(y_hi - y_lo)

def candidate_keyframes(salience_curve):
    return np.argwhere(salience_curve > np.mean(salience_curve))

def cluster_keyframes_simple(keyframes, thresh=3):
    itr = 0
    cls = []
    end = len(keyframes)
    for i in range(0, end - 1):
        d = keyframes[i + 1] - keyframes[i]
        if d > thresh:
            cls.append(keyframes[itr:i+1])
            itr = i+1
        
    cls.append(keyframes[itr:end])
    return cls

def cluster_keyframes_kmeans(keyframes, n_clusters):
    to_cluster = keyframes
    dummy_axis = np.arange(len(keyframes))
    X = np.vstack((to_cluster, dummy_axis)).T
    
    kmeans = KMeans(n_clusters=n_clusters, init='random')
    kmeans.fit(X)
    
    cls = [[] for _ in range(n_clusters)]
    for k, l in zip(keyframes, kmeans.labels_):
        cls[l].append(k)
    
    return X, kmeans.labels_, kmeans.cluster_centers_, cls

def nearest_center_kmeans(samples, labels, centers):
    
    nearest = np.zeros_like(centers)
    for idx, cen in enumerate(centers):
        _slice = np.argwhere(labels == idx).flatten()
        _samples = samples[_slice]
        _dists = length(_samples - cen)
        nearest[idx] = _samples[np.argmin(_dists)]
    
    return np.sort(nearest[:, 0])
        
def decimate_keyframes(kfcls, orig_curve):
    kf = []
    curve = orig_curve - np.mean(orig_curve)
    for c in kfcls:
        lmin = np.abs(np.min(curve[c]))
        lmax = np.abs(np.max(curve[c]))

        idx = 0
        if lmin > lmax:
            idx = np.argmin(curve[c])
        else:
            idx = np.argmax(curve[c])

        kf.append(c[idx])
    
    kf.sort()
    return kf

def lpf_motion():
    b = np.array([0.1526249789, 0.0333481282, 0.0777551903, 0.0667145281, 0.0138945068])
    a = np.array([1, -1.7462227354, 1.7354077932, -0.8232679111, 0.1793463694])
    return (b, a)

def lpf_motion_d1():
    b = np.array([0.1973679432, -0.0056567353, -0.0321850947, -0.1099445540, -0.0495815592])
    a = np.array([1, -0.9870779094, 0.7774863652, -0.2206843188, 0.02813441289])
    return (b, a)
    
def lpf_motion_d2():
    b = np.array([-0.0795571277, 0.1390709784, -0.0479192600, -0.0031459045, -0.0084486862])
    a = np.array([1, -1.571029458, 1.459212744, -0.7173743414, 0.1488005975])
    return (b, a)

def filter_motion(b, a, signal, dims):
    out_signals = []
    for dim in range(0, dims):
        x = signal[:, dim]
        y = lfilter(b, a, x)
        out_signals.append(y)
        
    return np.vstack((tuple(out_signals))).T

def lfderivator(coefs, filter_func, data_dict, dims=2):
    return {
        k:filter_func(coefs[0], coefs[1], p, dims) \
            for k, p in data_dict.items()
    }

def dict_operator(op_func=lambda x: x, in_dict={}):
    return {k:op_func(p) for k, p in in_dict.items()}

def dict2mat(in_dict, for_keys=None, transpose=True):
    out_mat = None
    if not for_keys:
        out_mat = np.asarray(list(in_dict.values()))
    else:
        out_mat = np.array([in_dict[k] for k in for_keys])
    
    return out_mat.T if transpose else out_mat

def velocity(x, dt=1.):
    v = x[2:] - x[0:-2]
    v_pad = np.pad(v, (1, 1), 'edge')
    return np.divide(v_pad, 2.*dt)

def acceleration(x, dt=1.):
    a = x[2:] - 2.*x[1:-1] + x[0:-2]
    a_pad = np.pad(a, (1, 1), 'edge')
    return np.divide(a_pad, dt**2)

def jerk(x, dt=1.):
    j = x[4:] - 2.*x[3:-1] + 2.*x[0:-4] - x[1:-3]
    j_pad = np.pad(j, (2, 2), 'edge')
    return np.divide(j_pad, 2.*dt**3)

def filter_motion2(x, func, fs=1.):
    out_signals = []
    for col in range(x.shape[1]):
        y = func(x[:, col], fs)
        out_signals.append(y)
        
    return np.vstack(tuple(out_signals)).T

def rawderivator(func, data_dict, fs=1.):
     return {
        k:filter_motion2(p, func, fs) for k, p in data_dict.items()
    }   

class Gesture(object):
    
    effort_keys = ['weight', 'time', 'space', 'flow', 'shape']
    
    weight_table = {
        'weight': np.array([0.1, 0.3, 0.3, 0.15, 0.15]),
        'time': np.array([0.3, 0.25, 0.25, 0.1, 0.1]),
        'space': np.array([0.1, 0.225, 0.225, 0.225, 0.225]),
        'flow': np.array([0.1, 0.225, 0.225, 0.225, 0.225]),
        'shape': np.array([0.1, 0.225, 0.225, 0.225, 0.225])
    }
    
    effector_table = {
        'weight': ['head', 'leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle'],
        'time': ['head', 'leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle'],
        'space': ['head', 'leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle'],
        'flow': ['head', 'leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle'],
        'shape': ['head', 'leftWrist', 'rightWrist', 'leftAnkle', 'rightAnkle']
    }
    
    def __init__(self, laban_obj):
        
        self.callback_table = {
            'time': self.aggregate_effort,
            'flow': self.aggregate_effort,
            'shape': self.aggregate_effort,
            'weight': self.weight_effort,
            'space': self.space_effort
        }
        
        self.trajectory_table = {
            'weight': laban_obj.joint_speed,
            'time': laban_obj.joint_mgacc,
            'space': laban_obj.joint_pos,
            'flow': laban_obj.joint_mgjrk,
            'shape': laban_obj.joint_curvature
        }
        
        self.__keyframes = laban_obj._LabanApproximator__keyframes
        self.ngestures = len(self.__keyframes) - 1
        self.effort_table = np.zeros((self.ngestures, len(Gesture.effort_keys)))
        
        for idx, key in enumerate(Gesture.effort_keys):
            t = self.trajectory_table[key]
            e = Gesture.effector_table[key]
            w = Gesture.weight_table[key]
            cb = self.callback_table[key]
            self.__push_effort(t, e, w, cb, self.effort_table[:, idx])
        
    def get_motion_features(self):
        return self.effort_table
    
    def aggregate_effort(self, segment_matrix, weights):
        T = segment_matrix.shape[0]
        mat_sum = np.sum(segment_matrix, axis=0) / T
        return np.dot(mat_sum, weights)
    
    def weight_effort(self, segment_matrix, weights):
        mat_sqr = np.power(segment_matrix, 2.)
        mat_w = np.multiply(mat_sqr, weights)
        mat_sum = np.sum(mat_w, axis=1)
        return np.amax(mat_sum)
    
    def space_effort(self, segment_matrix, weights):
        tot_displacement2d = segment_matrix[:, 1:] - segment_matrix[:, 0:-1]
        tot_displacement1d = length(tot_displacement2d, axis=2)
        tot_disp_sum = np.sum(tot_displacement1d, axis=1)
        displacement2d = segment_matrix[:, -1] - segment_matrix[:, 0]
        displacement1d = length(displacement2d)
        disp_ratio = np.divide(displacement1d, tot_disp_sum + 0.001)
        return np.dot(disp_ratio, weights)
        
    def __push_effort(self, mfeature, jointseg, weights, effort_func, out_arr):
        for idx, segment in enumerate(self.__iseg(mfeature, jointseg)):
            out_arr[idx] = effort_func(segment, weights)
        
    def __iseg(self, mfeature, jointseg):
        isnd = len(mfeature['head'].shape) > 1
        fmat = dict2mat(mfeature, jointseg, not isnd)
        for i in range(0, len(self.__keyframes) - 1):
            lo = self.__keyframes[i]
            hi = self.__keyframes[i + 1]
            yield fmat[:, lo:hi] if isnd else fmat[lo:hi, :]
     
    
class LabanApproximator(object):
    
    __H_PAIRS = [
        ('rightShoulder', 'rightElbow'), ('rightElbow', 'rightWrist'),
        ('leftShoulder', 'leftElbow'), ('leftElbow', 'leftWrist'),
        ('rightHip', 'rightKnee'), ('rightKnee', 'rightAnkle'),
        ('leftHip', 'leftKnee'), ('leftKnee', 'leftAnkle')
    ]
    
    ALL_SCALE = None
    GESTURES_PER_BATCH = 7
    
    def __init__(self, lma_segment):
        
        data = lma_segment.joint_data
        if LabanApproximator.ALL_SCALE:
            data = dict_operator(self.world_scale, data)
        
        self.joint_pos = lfderivator(lpf_motion(), filter_motion, data)
        
        fs = 1. / lma_segment.meta_data['frameRate']        
        self.joint_vel = rawderivator(velocity, self.joint_pos, fs)
        self.joint_acc = rawderivator(acceleration, self.joint_pos, fs)
        self.joint_jrk = rawderivator(jerk, self.joint_pos, fs)
        
        self.joint_curvature = {
            k:curvature(vp, ap) for (k, vp), (_, ap)\
                in zip(self.joint_vel.items(), self.joint_acc.items())
        }
        
        self.joint_speed = dict_operator(length, self.joint_vel)
        self.joint_mgacc = dict_operator(length, self.joint_acc)
        self.joint_mgjrk = dict_operator(length, self.joint_jrk)
        self.bone_angles = self.bone_angle_array() 
        
        self.wlen = 5
        self.sig  = 0.75
        self.cluster_dist1 = self.wlen
        self.lin_weights = np.ones((self.bone_angles.shape[1], )) / self.bone_angles.shape[1]
        
        self.__keyframes, self.__rcurve = self.keyframes()
        self.__gesture = Gesture(self)
        

    def get_motion_features(self):
        return self.__gesture.get_motion_features()
    
    def world_scale(self, array):
        return np.multiply(array, LabanApproximator.ALL_SCALE)
    
    def bone_angle_array(self):
        p = self.joint_pos
        root = np.array([0, 0])
        clav = (p['leftHip'] + p['rightHip']) / 2.
        center_bone = clav - root
                
        return np.asarray([
            angle_btw(v, center_bone) for v in [
                p[bp[1]] - p[bp[0]] for bp in LabanApproximator.__H_PAIRS
            ]
        ]).T
        
    def keyframes(self):
        keyframes = []
        a = self.bone_angles
        for col in range(0, a.shape[1]):
            curve = a[:, col]
            salience = curve_salience(curve, self.wlen, self.sig)
            candidkf = candidate_keyframes(salience)
            kcluster = cluster_keyframes_simple(candidkf, self.cluster_dist1)
            keyframes.append(decimate_keyframes(kcluster, curve))

        k = reduce(np.union1d, tuple(keyframes))
        _, _, _, cls = cluster_keyframes_kmeans(k, n_clusters=LabanApproximator.GESTURES_PER_BATCH + 1)
        ref_curve = lincomb(a, self.lin_weights)
        out_keyframes = decimate_keyframes(cls, ref_curve)
        return (out_keyframes, ref_curve)
    
    def plt_keyframes(self, fignum=0, figsize=(10, 8)):
        fig, axs = plt.subplots(2, 1, num=fignum, figsize=figsize)
        pp.plt_keyframe_curve(axs[0], self.__keyframes, self.__rcurve)
        pp.plt_keyframe_skeleton(axs[1], self.__keyframes, self.joint_pos, 200.)
        plt.show()


class LMARunner(object):
    
    def __init__(self, video_segments):
        
        self.lma_list = []
        feature_list = []
        
        tot = len(video_segments) - 1
        for elp, seg in enumerate(video_segments):
            lma = LabanApproximator(seg)
            self.lma_list.append(lma)
            feature = lma.get_motion_features()
            feature_list.append(feature)
            
            if print_progress:
                print_progress('[*]  Processing', elp/tot)
        
        self.nframes = len(self.lma_list)
        self.ngestures = LabanApproximator.GESTURES_PER_BATCH 
        self.motion_features_raw = np.vstack(tuple(feature_list))
            
    def dataframe(self):
        cols = Gesture.effort_keys
        return pd.DataFrame(self.motion_features_raw, columns=cols)
    
    def plot_keyframes_for(self, frame_num, fignum=0):
        self.lma_list[frame_num].plt_keyframes(fignum=fignum)
    
    @staticmethod
    def set_lma_hyperparams(scale=1, gestures_per_batch=7, effectors_weights=None):
        LabanApproximator.ALL_SCALE = scale
        LabanApproximator.GESTURES_PER_BATCH = gestures_per_batch
        
        if not effectors_weights:
            return
        
        for bea, e, w in effectors_weights:
            Gesture.effector_table[bea] = e
            Gesture.weight_table[bea] = w
