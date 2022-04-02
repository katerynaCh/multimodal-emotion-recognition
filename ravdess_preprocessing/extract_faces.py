# -*- coding: utf-8 -*-
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(image_size=(720, 1280), device=device)

#mtcnn.to(device)
save_frames = 15
input_fps = 30

save_length = 3.6 #seconds
save_avi = True

failed_videos = []
root = '/lustre/scratch/chumache/RAVDESS_or/'

select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0
for sess in tqdm(sorted(os.listdir(root))):   
    for filename in os.listdir(os.path.join(root, sess)):
           
        if filename.endswith('.mp4'):
                        
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))  
            #calculate length in frames
            framen = 0
            while True:
                i,q = cap.read()
                if not i:
                    break
                framen += 1
            cap = cv2.VideoCapture(os.path.join(root, sess, filename))

            if save_length*input_fps > framen:                    
                skip_begin = int((framen - (save_length*input_fps)) // 2)
                for i in range(skip_begin):
                    _, im = cap.read() 
                    
            framen = int(save_length*input_fps)    
            frames_to_select = select_distributed(save_frames,framen)
            save_fps = save_frames // (framen // input_fps) 
            if save_avi:
                out = cv2.VideoWriter(os.path.join(root, sess, filename[:-4]+'_facecroppad.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))

            numpy_video = []
            success = 0
            frame_ctr = 0
            
            while True: 
                ret, im = cap.read()
                if not ret:
                    break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((sess, i))
                    break
	    
                temp = im[:,:,-1]
                im_rgb = im.copy()
                im_rgb[:,:,-1] = im_rgb[:,:,0]
                im_rgb[:,:,0] = temp
                im_rgb = torch.tensor(im_rgb)
                im_rgb = im_rgb.to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                im = im[y1:y2, x1:x2, :]
                im = cv2.resize(im, (224,224))
                if save_avi:
                    out.write(im)
                numpy_video.append(im)
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    if save_avi:
                        out.write(np.zeros((224,224,3), dtype = np.uint8))
                    numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
            if save_avi:
                out.release() 
            np.save(os.path.join(root, sess, filename[:-4]+'_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error', sess, filename)    
                            
    n_processed += 1      
    with open('processed.txt', 'a') as f:
        f.write(sess + '\n')
    print(failed_videos)
