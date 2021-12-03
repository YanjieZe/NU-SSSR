import cv2


video_paths = ["../real_video/%u.mkv"%i for i in range(1,6)]
count = 1
time_gap = 0
for video_path in video_paths:
    print('new video is readed.')
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    
    while success:
        time_gap += 1
        
          
        success,image = vidcap.read()
        if(time_gap%200!=0):
            continue
        cv2.imwrite("../real_img/%d.jpg" % count, image)     # save frame as JPEG file    
        print('Read a new frame: ', success)
        count += 1