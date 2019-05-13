import cv2
import dlib
import time


 

if __name__ == '__main__' :
    detector = dlib.get_frontal_face_detector()
    # Start default camera
    video = cv2.VideoCapture(0)
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
     
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    #video.set(cv2.CAP_PROP_FPS,30)
    #video.set(cv2.CAP_PROP_EXPOSURE,500) 
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
        exp = video.get(cv2.CAP_PROP_EXPOSURE)
        print "Exposur: {0}".format(exp)
     
 
    # Number of frames to capture
    num_frames = 500
     
     
    print "Capturing {0} frames".format(num_frames)
 
    # Start time
    start = time.time()
     
    # Grab a few frames

    #for i in xrange(0, num_frames) :
    #    ret, frame = video.read()
    
    
    for i in xrange(0, num_frames) :
        ret, frame = video.read()
        cv2.imshow('frame',frame)

        #dets = detector(frame, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
     
    # End time
    end = time.time()
 
    # Time elapsed
    seconds = end - start
    print "Time taken : {0} seconds".format(seconds)
 
    # Calculate frames per second
    fps  = num_frames / seconds
    print "Estimated frames per second : {0}".format(fps)
 
    # Release video
    video.release()

