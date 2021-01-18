#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import numpy as np
from scipy import signal
import math
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt
import multiprocessing as mp

#===== Constants =====
DIRECTORY = 'imgtest'
DIRECTION = 'left'
#Feature
FEATURE_THRESHOLD = 0.01
FEATURE_EDGE = 30
EDGE = 30
CONST_K = 0.04
KERNEL = 5
WINDOW = 4
#Matchimg
Y_MATCH = 15
RANSAC_K = 1000
RANSAC_THRESHOLD_DISTANCE = 3
#Blending
ALPHA_BLEND_WINDOW = 60
#Save a bunch a files during the process
SAVE = True
#===== Constants =====

#Pre-work
def ReadImages():
    filename = DIRECTORY + '/readImage2a.txt'
    open_input = open(filename, 'r')
    imagefiles = []
    focal_length = []
    for img in open_input:
        if img[0] =='#':
            continue
        (imagename, focal) = img.split()
        imagefiles += [imagename]
        focal_length += [focal]
    images = [cv2.imread(DIRECTORY + "/" + i) for i in imagefiles]
    
    images = np.array(images)
    focal_length = np.array(focal_length, dtype = np.float32)
    
    open_input.close()
    
    return images, focal_length

def Projection(img, focal_length):
    height, width, _ = img.shape
    cylinder = np.zeros(shape = img.shape, dtype = np.uint16)
    for i in range(-int(height / 2), int(height / 2)):
        for j in range(-int(width / 2), int(width / 2)):
            cylinder_x = focal_length * math.atan(j / focal_length)
            cylinder_x = int(round(cylinder_x + width / 2))
            
            cylinder_y = focal_length * (i / math.sqrt(j**2 + focal_length**2))
            cylinder_y = int(round(cylinder_y + height / 2))
            
            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder[cylinder_y][cylinder_x] = img[i + int(height / 2)][j + int(width / 2)]
    
    return cylinder

#Feature detection(Harris)
def FeatureDetection(img, pool, window_size = WINDOW):
    k = CONST_K

    #這是python裡面自帶的找harris corner的函式
    # newImg = cv2.cornerHarris(gray, 2, 3, 0.04)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) 
    
    response = np.zeros(shape = gray.shape, dtype = np.float32)
    
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    #Subtasks
    Sx2 = cv2.boxFilter(Ix2, -1, (window_size, window_size), normalize = False)
    Sy2 = cv2.boxFilter(Iy2, -1, (window_size, window_size), normalize = False)
    Sxy = cv2.boxFilter(Ixy, -1, (window_size, window_size), normalize = False)
    
    height, width, _ = img.shape
    
    response = pool.starmap(ComputeR, [(Sx2[i], Sy2[i], Sxy[i], k) for i in range(height)])
    response = np.asarray(response)
    
    return response

def ComputeR(Sx2, Sy2, Sxy, k):
    response = np.zeros(shape = Sx2.shape, dtype = np.float32)
    
    for i in range(len(Sx2)):
        det = Sx2[i] * Sy2[i] - (Sxy[i]**2)
        trace = Sx2[i] + Sy2[i]
        R = det - k * (trace**2)
        response[i] = R
        
    return response

#Feature description()
def FeatureDescription(img, corner_response, threshold = FEATURE_THRESHOLD, edge = FEATURE_EDGE, window_size = WINDOW):
    #Pick response bigger than a threshold
    height, width = corner_response.shape
    features = np.zeros(shape = (height, width), dtype = np.uint16)
    
    #Supression
    features[corner_response > threshold * corner_response.max()] = 255
    features[     : EDGE,      :     ] = 0
    features[-EDGE:     ,      :     ] = 0
    features[     :     , -EDGE:     ] = 0
    features[     :     ,      : EDGE] = 0
    
    for i in range(0, height, window_size):
        for j in range(0, width, window_size):
            if features[i: i + window_size, j: j + window_size].sum() == 0:
                continue
            else:
                block = corner_response[i: i + window_size, j: j + window_size]
                max_i, max_j = np.unravel_index(np.argmax(block), (window_size, window_size))

                features[i: i + window_size, j: j + window_size] = 0
                features[i + max_i][j + max_j] = 255
            
    kernel_size, half_kernel = KERNEL, KERNEL // 2
    feature_pos, feature_des = [], np.zeros(shape = (1, kernel_size**2), dtype = np.float32)
    for i in range(half_kernel, height - half_kernel):
        for j in range(half_kernel, width - half_kernel):
            if features[i][j] == 255:
                desc = corner_response[i - half_kernel: i + half_kernel + 1, j - half_kernel: j + half_kernel + 1]
                feature_pos += [[i, j]]
                feature_des = np.append(feature_des, [desc.flatten()], axis = 0)
                
    return feature_des[1: ], feature_pos

#Feature matching
def FeatureMatching(des1, des2, pos1, pos2, pool):
    matched_pairs = []
    sub_des, sub_pos = np.array_split(des1, 64), np.array_split(pos1, 64)
    
    sub_tasks = [(sub_des[i], des2, sub_pos[i], pos2) for i in range(64)]
    results = pool.starmap(ComputeM, sub_tasks)
    
    for r in results:
        if len(r) > 0:
            matched_pairs += r
    
    return matched_pairs

def ComputeM(des1, des2, pos1, pos2):
    matched_pairs = []
    matched_pairs_rank = []
    
    for i in range(len(des1)):
        dist = []
        y = pos1[i][0]
        for j in range(len(des2)):
            diff = float('INF')
            
            #Compare features that have simular y-axis and just get rid of those on the top edge. It's crazy.
            if (y - Y_MATCH <= pos2[j][0] <= y + Y_MATCH) and y > 100:
                diff = ((des1[i] - des2[j])**2).sum()
            dist += [diff]
        sorted_index = np.argpartition(dist, 1)
        local_optimal, second_optimal = dist[sorted_index[0]], dist[sorted_index[1]]
        if local_optimal > second_optimal:
            local_optimal, second_optimal = second_optimal, local_optimal
            
        if local_optimal / second_optimal <= 0.8:
            index = np.where(dist == local_optimal)[0][0]
            matched_pairs += [[pos1[i], pos2[index]]]
            matched_pairs_rank += [local_optimal]
            
    sorted_rank_index = np.argsort(matched_pairs_rank)
    sorted_match_pairs = np.asarray(matched_pairs)
    sorted_match_pairs = sorted_match_pairs[sorted_rank_index]
    
    refined_matched_pairs = []
    for i in sorted_match_pairs:
        duplicated = False
        for r in refined_matched_pairs:
            if(r[1] == list(i[1])):
                duplicated = True
                break
        if not duplicated:
            refined_matched_pairs += [i.tolist()]
    return refined_matched_pairs

#Image matching(Stitching) and Blending
def RANSAC(matched_pairs, last_shift):
    matched_pairs = np.asarray(matched_pairs)
    
    threshold_distance = RANSAC_THRESHOLD_DISTANCE
    use_random = True if len(matched_pairs) > RANSAC_K else False
    K = RANSAC_K if use_random else len(matched_pairs)
    
    best_shift = []
    max_inliner = 0
    for k in range(K):
        inliner = 0
        
        index = int(np.random.random_sample() * len(matched_pairs)) if use_random else k
        sample = matched_pairs[index]
        
        shift = sample[1] - sample[0]
        
        shifted = matched_pairs[:, 1] - shift
        difference = matched_pairs[:, 0] - shifted
        
        for d in difference:
            if np.sqrt((d**2).sum()) < threshold_distance:
                inliner += 1
                
        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = shift
            
    return best_shift

def ImageStiching(img1, img2, shift, pool):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shift1 = np.lib.pad(img1, padding, 'constant', constant_values = 0)
    
    #Cut image
    split = img2.shape[1] + abs(shift[1])
    splited = shift1[:, split:] if shift[1] > 0 else shift1[:, :-split]
    shift1 = shift1[:, :split] if shift[1] > 0 else shift1[:, -split:]
    
    height1, width1, _ = shift1.shape
    height2, width2, _ = img2.shape
    
    inv_shift = [height1 - height2, width1 - width2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shift2 = np.lib.pad(img2, inv_padding, 'constant', constant_values = 0)
    
    #Blending
    seam_x = shift1.shape[1] // 2
    sub_tasks = [(shift1[i], shift2[i], seam_x, ALPHA_BLEND_WINDOW, DIRECTION) for i in range(height1)]
    shift1 = pool.starmap(Blending, sub_tasks)
    
    shift1 = np.asarray(shift1)
    shift1 = np.concatenate((shift1, splited) if shift[1] > 0 else (splited, shift1), axis = 1)
    
    return shift1

def Blending(r1, r2, seam_x, window_size, direction):
    if direction == 'right':
        r1, r2 = r2, r1
        
    blend_row = np.zeros(shape = r1.shape, dtype = np.uint16)
    for i in range(len(r1)):
        color1 = r1[i]
        color2 = r2[i]
        if i < seam_x - window_size:
            blend_row[i] = color2
        elif i > seam_x + window_size:
            blend_row[i] = color1
        else:
            alpha = (i - seam_x + window_size) / (window_size * 2)
            blend_row[i] = (1 - alpha) * color2 + alpha * color1
            
    return blend_row

#End to end alignment
def Alignment(img, shifts):
    sum_x, sum_y = np.sum(shifts, axis = 0)
    
    col_shift = None
    
    #/-way or \-way
    if sum_x * sum_y > 0:
        col_shift = np.linspace(np.abs(sum_x), 0, num = img.shape[1], dtype = np.uint16)
    else:
        col_shift = np.linspace(0, xnp.abs(sum_x), num = img.shape[1], dtype = np.uint16)
    aligned = img.copy()
    for i in range(img.shape[1]):
        aligned[:, i] = np.roll(img[:, i], col_shift[i], axis = 0)
    return aligned


# In[2]:


img, focallist = ReadImages()
plt.figure(figsize = (20, 40))
for i in range(len(img)):
    imgplt = plt.subplot(len(img), len(img), i + 1)
    imgplt.imshow(img[i])


# In[3]:


pool = mp.Pool(mp.cpu_count())
cylinder_img = pool.starmap(Projection, [(img[i], focallist[i]) for i in range(len(img))])
#cylinder_img = img
plt.figure(figsize = (20, 40))
for i in range(len(img)):
    imgplt = plt.subplot(len(img), len(img), i + 1)
    if SAVE:
        output = DIRECTORY + "/dsttest/warp" + str(i) + '.JPG'
        cv2.imwrite(output, cylinder_img[i])
    imgplt.imshow(cylinder_img[i])
stitched_image = cylinder_img[0].copy()
height, width, _ = img[0].shape
shifts = [[0, 0]]
last_feature = [[], []]


# In[4]:


for i in range(1, len(img)):
    img1 = cylinder_img[i - 1]
    img2 = cylinder_img[i]
    
    des1, pos1 = last_feature
    
    if i == 1:
        res1 = FeatureDetection(img1, pool)
        des1, pos1 = FeatureDescription(img1, res1)
        fimg = img1.copy()
        for (x, y) in pos1:
            fimg.itemset((x, y, 0), 0)
            fimg.itemset((x, y, 1), 0)
            fimg.itemset((x, y, 2), 255)
        if SAVE:
            output = DIRECTORY + "/dsttest/" + str(i - 1) + '.JPG'
            cv2.imwrite(output, fimg)
    
    res2 = FeatureDetection(img2, pool)
    des2, pos2 = FeatureDescription(img2, res2)
    
    fimg = img2.copy()
    for (x, y) in pos2:
        fimg.itemset((x, y, 0), 0)
        fimg.itemset((x, y, 1), 0)
        fimg.itemset((x, y, 2), 255)
    if SAVE:
        output = DIRECTORY + "/dsttest/" + str(i) + '.JPG'
        cv2.imwrite(output, fimg)
    
    matched_pairs = FeatureMatching(des1, des2, pos1, pos2, pool)
    _, offset, _ = img1.shape
    plt_img = np.concatenate((img1, img2), axis = 1)
    plt.figure(figsize = (20, 40))
    plt.imshow(plt_img)
    for j in range(len(matched_pairs)):
        plt.scatter(x = matched_pairs[j][0][1], y = matched_pairs[j][0][0], c = 'r')
        plt.plot([matched_pairs[j][0][1], offset + matched_pairs[j][1][1]], [matched_pairs[j][0][0], matched_pairs[j][1][0]], 'y-', lw = 0.7)
        plt.scatter(x = offset + matched_pairs[j][1][1], y = matched_pairs[j][1][0], c = 'r')
    plt.show()
    
    shift = RANSAC(matched_pairs, shifts[-1])
    shifts += [shift]
    
    stitched_image = ImageStiching(stitched_image, img2, shift, pool)
    if SAVE:
        output = DIRECTORY + "/stitchtest/" + str(i) + '.JPG'
        cv2.imwrite(output, stitched_image)
    last_feature = [des2, pos2]


# In[5]:


aligned = Alignment(stitched_image, shifts)
plt.figure(figsize = (20, 40))
plt.imshow(aligned)
output = DIRECTORY + "/Result/Alligned.JPG"
cv2.imwrite(output, aligned)


# In[6]:


#Crop the image
_, threshold = cv2.threshold(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
upper, lower = [-1, -1]
cut_threshold = aligned.shape[1] * 0.025

for i in range(threshold.shape[0]):
    if len(np.where(threshold[i] == 0)[0]) < cut_threshold:
        upper = i
        break
for i in range(threshold.shape[0] - 1, 0, -1):
    if len(np.where(threshold[i] == 0)[0]) < cut_threshold:
        lower = i
        break
        
cropped = aligned[upper: lower, :]
plt.imshow(cropped)
output = DIRECTORY + "/Result/Cropped.JPG"
cv2.imwrite(output, cropped)

