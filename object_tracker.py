import os
import time
import pickle
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

flags.DEFINE_boolean('load_checkpoint', True, 'load inference checkpoint')
flags.DEFINE_string('inference_checkpoint', 'checkpoints/inference-checkpoint.pickle', 'path to inference checkpoint pickle')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = 10
    nms_max_overlap = 1.0

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    video_path = FLAGS.video


    out = None

    load_checkpoint   = FLAGS.load_checkpoint
    checkpoint_file   = FLAGS.inference_checkpoint
    inference_results=[]

    if load_checkpoint:
        with open(checkpoint_file, 'rb') as f:
            inference_results = pickle.load(f)

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    total_frames=len(inference_results)

    frame_num = 0
    fps_sum=0
    # while video is running
    while True:
        if frame_num==total_frames:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num +=1
        print('\nFrame %3d:' % frame_num)
        start_time = time.time()

        if not load_checkpoint:
            print("\n\nERROR: This program only works with checkpoint files!\n")
            break
        else:
            # Get current values from checkpoint
            infres = inference_results[frame_num - 1]
            [bboxes_list, scores, classes, num_objects, features, detections] = infres
            bboxes = np.asarray(bboxes_list)


        #initialize color map
        #cmap = plt.get_cmap('tab20b')
        #colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        st = time.time()
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        elapsed = time.time() - st
        print(' Non-maxima suppression: %4.2f ms' % (elapsed*1000))

        # Call the tracker
        st = time.time()
        tracker.predict()
        elapsed = time.time() - st
        print(' KF predict: %3.1f ms' % (elapsed*1000))

        print(' Tracker update:')
        st = time.time()
        tracker.update(detections)
        elapsed = time.time() - st
        print(' Update total: %4.1f ms' % (elapsed*1000))

        # update tracks
        #for track in tracker.tracks:
            #if not track.is_confirmed() or track.time_since_update > 1:
            #    continue
            #bbox = track.to_tlbr()
            #class_name = track.get_class()

            # draw bbox on screen
            #color = colors[int(track.track_id) % len(colors)]
            #color = [i * 255 for i in color]

            # if enable info flag then print details about each track
            #if FLAGS.info:
            #    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        elapsed = time.time() - start_time
        fps = 1.0 / (elapsed)
        print("Total: %5.1f ms" % (elapsed*1000), end="  ")
        print("/ %5.2f" % fps, "FPS")
        fps_sum += fps

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("Average FPS: %.2f" % (fps_sum / frame_num))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
