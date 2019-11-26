meta_keys = [
         'videoStart', 'videoEnd', 'totalFrames',
         'frameRate', 'skipped', 'maxSpan'
]

joint_keys = [
        'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
        'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
        'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle',
        'head', 'torso'
]

joint_indeces = [
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16
]

template = "\r{0}: [{1:50s}] {2:.1f}%"
def print_progress(description, amount, marker='#'):
    print(
          template.format(description, marker * int(amount * 50), amount*100), end="", flush=True
    )

from .LMAJSONUtil import LMAJSONUtil
from .LMAApproximator import LMARunner
