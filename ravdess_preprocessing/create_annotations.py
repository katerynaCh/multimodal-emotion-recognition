# -*- coding: utf-8 -*-

import os
root = '/lustre/scratch/chumache/RAVDESS_or/'
#splits used in the paper with 5 folds
#n_folds=5
#folds =[ [[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]],[[5,6,7,8],[9,10,11,12],[13,14,15,16,17,18,19,20,21,22,23,24,1,2,3,4]],[[9,10,11,12],[13,14,15,16],[17,18,9,20,21,22,23,24,1,2,3,4,5,6,7,8]],[[13,14,15,16],[17,18,19,20],[21,22,23,24,1,2,3,4,5,6,7,8,9,10,11,12]],[[17,18,19,20],[21,22,23,24],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]]

n_folds=1
folds = [[[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]]
for fold in range(n_folds):
        fold_ids = folds[fold]
        test_ids, val_ids, train_ids = fold_ids
	
        #annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
        annotation_file = 'annotations.txt'
	
        for i,actor in enumerate(os.listdir(root)):
            for video in os.listdir(os.path.join(root, actor)):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
		     
                audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'  
                if i in train_ids:
                   with open(annotation_file, 'a') as f:
                       f.write(os.path.join(root,actor, video) + ';' + os.path.join(root,actor, audio) + ';' + label + ';training' + '\n')
		

                elif i in val_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';validation' + '\n')
		
                else:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';testing' + '\n')
		

