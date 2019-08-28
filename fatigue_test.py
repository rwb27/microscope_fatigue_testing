
from __future__ import print_function
import io
import sys
import os
import time
import argparse
import numpy as np
import picamera
from builtins import input
from readchar import readchar, readkey
from openflexure_stage import OpenFlexureStage
from openflexure_microscope import load_microscope
from openflexure_microscope.microscope import picamera_supports_lens_shading
import scipy
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from contextlib import contextmanager, closing
import data_file
import cv2
from camera_stuff import find_template
#import h5py
import threading
import queue

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "fatigue_tests.hdf5")) as df:
        
        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        step_size = 1000
        n_steps = 20
        backlash = 0

        camera.resolution=(1640,1232)
        stage.backlash = backlash

        data_group = df.new_group("fatigue_test")
        data_group.attrs['step_size'] = step_size
        data_group.attrs['n_steps'] = n_steps
        data_group.attrs['backlash'] = backlash
        data_group.attrs['camera_resolution'] = np.array(camera.resolution)

        camera.start_preview(resolution=(640,480))

        #stage.move_rel([-backlash, -backlash, -backlash])
        #stage.move_rel([backlash, backlash, backlash])
        data_group['initial_sample_image'] = ms.rgb_image()

        frame = ms.rgb_image().astype(np.float32).mean(axis=2)
        #mean = np.mean(image)
        #templ8 = (image - mean)[100:-100, 100:-100]
        template = np.load("template.npz")['template']
        
        data_group['template_image'] = template

        d, corr = find_template(template, frame, return_corr = True, fraction=0.15)        

        initial_stage_position = stage.position
        start_time = time.time()
        data_group.attrs['start_time'] = start_time
        last_saved_image_time = 0

        try:
            while True:
                for d in [1, 0, -1, 0]:
                    data_stage = np.zeros((n_steps + 1, 3))
                    data_cam = np.zeros((n_steps + 1, 2))
                    data_time = np.zeros((n_steps+1,))
                    for i in range(n_steps+1):
                        if i>0:
                            stage.move_rel([d*step_size, 0, 0])
                        data_stage[i, 0:] = stage.position
                        frame = ms.rgb_image().astype(np.float32).mean(axis=2)
                        data_cam[i, 0:], corr = find_template(template, frame, return_corr = True, fraction=0.15)
                        data_time[i] = time.time()
                    df.add_data(data_stage, data_group, "data_stage")
                    df.add_data(data_cam, data_group, "data_cam")
                    df.add_data(data_time, data_group, "data_time")
                if time.time() - last_saved_image_time > 900:
                    # periodically save a JPEG of what the camera sees.  NB we could do this more
                    # efficiently by capturing JPEG directly - but it's hardly time critical.
                    frame = ms.rgb_image()
                    df.add_data(np.array(cv2.imencode('.jpg', frame)[1]), data_group, "image")
                    last_saved_image_time = time.time()
        except KeyboardInterrupt:
            print("Got a keyboard interrupt, stopping...")

        stage.move_abs(initial_stage_position)
        camera.stop_preview()
        stage.release_motors()
        print("Done")
