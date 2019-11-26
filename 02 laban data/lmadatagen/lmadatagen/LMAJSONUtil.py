#!/usr/bin/env python
# coding: utf-8

import json
from json import JSONDecodeError
from socket import error, gaierror, herror, timeout
import urllib.request as urlrequest
from urllib.request import URLError
from pathlib import Path
import numpy as np
from lmadatagen import print_progress

class LMAVideoSegment(object):
    
    def __init__(self, joint_data=None, meta_data=None, joint_keys=None):
        self.joint_data = joint_data
        self.meta_data = meta_data
        self.joint_keys = joint_keys
        
    def recon(self, _dict):
        self.joint_data = {
            k:np.asarray(v, dtype=np.float) for k, v in _dict['joint_data'].items()
        }
        self.meta_data = _dict['meta_data']
        self.joint_keys = _dict['joint_keys']
    
class LMAJSONUtil(object):
    
    wheel = ['-', '-', '-', '\\', '\\', '\\', '|', '|', '|', '/', '/', '/']
    
    def __init__(self, joint_keys=None, joint_indeces=None, meta_keys=None):
        self.joint_keys = joint_keys
        self.joint_indeces = joint_indeces
        self.meta_keys = meta_keys
        
        self.__segments = []
    
    def __glob_json(self, relative_path, process_callback):
        p = Path(relative_path)
        json_files = list(p.glob('*.json'))
        tot = len(json_files) - 1
        for elp, file in enumerate(json_files):
            #w = LMAJSONUtil.wheel[int(fcounter % len(LMAJSONUtil.wheel))]
            #print('\r[*] Processing: {0}'.format(w), end='', flush=True)
            if print_progress:
                print_progress('[*] Processing', elp/tot)
            
            yield process_callback(relative_path + file.name)

    def __open_json(self, file):
        with open(file) as file:
            return json.load(file)
    
    def __coordgen(self, frame_array):
        for fidx, frame in enumerate(frame_array):
            if frame['interpolation']:
                k = list(frame.keys())
                ex = ['head', 'torso']
                if not all(el in k for el in ex):
                    print('Skipped frame {0}'.format(fidx))
                    continue
                
            yield ('head', frame['head'])
            yield ('torso', frame['torso'])
            
            coords = frame['data']
            for idx, key in zip(self.joint_indeces, self.joint_keys):
                yield (key, coords[idx])
                
    def __on_process_motion_json(self, file):
        json = self.__open_json(file)
        
        p = self.joint_keys
        q = [[] for n in range(len(p))]
        d = dict(zip(p, q))
        frames = json['frames']
        
        for c in self.__coordgen(frames):
            d[c[0]].append(c[1])
        
        return {
            'meta': {k:json[k] for k in self.meta_keys},
            'data': {k:np.asarray(v, dtype=np.float) for k, v in d.items()}
        }
    
    def __save_motion_json(self, file):
        json_string = str(json.dumps(
            [ob.__dict__ for ob in self.__segments], 
            default=lambda _obj: _obj.tolist()
        )).encode('utf-8')
        
        file_size_mb = len(json_string) * 10**-6
        if file_size_mb > 25:
            print('File too big')
            return
        
        with open(file, 'w', encoding='utf8') as f:
            f.write(json_string)
        
    def __process_motion_json(self, path_motion):
        for p in self.__glob_json(path_motion, self.__on_process_motion_json):
            seg = LMAVideoSegment(
                joint_data=p['data'],
                meta_data=p['meta'],
                joint_keys=self.joint_keys
            )
            self.__segments.append(seg)        

    def __marshall_manual(self, json_data):
        for data in json_loaded:
            _obj = LMAVideoSegment()
            _obj.recon(data)
            self.__segments.append(_obj)

    def __load_local(self, file):
        with open(file, 'r') as f:
            data_loaded = json.loads(f.read())
            self.__marshall_manual(data_loaded)

    def __load_remote(self, remote_url):
        with urlrequest.urlopen(remote_url) as url:
            data_loaded = json.loads(url.read().decode())
            self.__marshall_manual(data_loaded)

    def __try_load_motion_json(self, ret_info):
        location, path = ret_info
        print('Processing file from: [{0}] {1}'.format(location, path))
        if location is 'Local':
            try:
                self.__load_local(file=path)
            except (FileNotFoundError, ValueError) as e:
                print('error on reading file: {0}'.format(e))
                return False

        elif location is 'Remote':
            try:
                self.__load_remote(remote_url=path)
            except (URLError, ValueError, gaierror, error, herror, timeout) as e:
                print('error on reading URL: {0}'.format(e))
                return False
        
        else:
            print('No such place to fetch file')
            return False
        
        return True
    
    def save_json(self, save_file):
        self.__save_motion_json(file=save_file)
    
    def get_video_segments(self, path_motion=None, ret_info=None):
        ret = True
        if not ret_info:
            self.__process_motion_json(path_motion)
        else:
            ret = self.__try_load_motion_json(ret_info)
        
        if not ret:
            print('Did not load any data. Returning empty list')
            return []
        
        elp = len(self.__segments)
        print('\r[*]  Processed {0} files {1}'.format(elp + 1, ' '*50), end='', flush=True)
        return self.__segments 
