import os
import cv2
import numpy as np
from multiprocessing import Pool
# import matplotlib.pyplot as plt

def calculate_frame_diff(frames):
    """
    두 이미지 프레임 간의 픽셀 값 차이의 총합을 계산하는 함수
    
    Parameters:
        frames (numpy.ndarray): shape이 (N, 512, 512, 3)인 이미지 프레임들의 numpy array
    
    Returns:
        numpy.ndarray: shape이 (N-1,)인 픽셀 값 차이의 총합을 담은 numpy array
    """
    num_frames = frames.shape[0]
    
    # 픽셀 값 차이를 저장할 배열을 생성합니다.
    frame_diff_avg = np.zeros(num_frames - 1)
    
    for i in range(num_frames - 1):
        # 현재 프레임과 다음 프레임 간의 픽셀 값 차이를 계산하고 총합을 구합니다.
        frame_diff_avg[i] = np.average(np.abs(frames[i] - frames[i+1]))
        #frame_diff_avg[i] = np.average(np.multiply(frames[i], frames[i+1]))
    
    return frame_diff_avg

def find_indices_exceeding_threshold(arr, threshold):
    """
    특정 threshold를 넘는 값을 가지는 인덱스를 반환하는 함수
    
    Parameters:
        arr (numpy.ndarray): shape이 (N-1,)인 numpy array
        threshold (float): 찾고자 하는 threshold 값
    
    Returns:
        list: 특정 threshold를 넘는 값을 가지는 인덱스 리스트
    """
    exceeding_indices = np.where(np.abs(np.diff(arr)) > threshold)[0]
    return exceeding_indices


def process_subject(subjectID):
    path = f'./dataset/{subjectID}/image/{subjectID}'
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tif')]
    #files.sort()
    files = files[1:]  # remove sagittal image, this may remove non-sagittal frames, but let's omit this

    #print(f'Subject ID: {subjectID}')
    #print(f'Number of files: {len(files)}')

    imgs = np.zeros((len(files), 512, 512, 3)).astype(np.uint8)
    for idx, file in enumerate(files):
        img = cv2.imread(file)  #, flags=cv2.IMREAD_UNCHANGED)
        H, W, C = img.shape
        imgs[idx,256-H//2:256+H//2,256-W//2:256+W//2] = img
        # try:
        #     #imgs[idx,256-H//2:256+H//2,256-W//2:256+W//2] = img
        #     imgs.append(img)
        # except:
        #     print(idx)

    #print(f'Image shape: {np.shape(imgs)}')
    imgs = np.array(imgs)
    diff = calculate_frame_diff(imgs)
    indices = find_indices_exceeding_threshold(diff, 15)
    print(f'{subjectID}: {indices}')
    #print()

    if len(indices) == 0:
        print("!!!!!!!!!!!!! REDUCE THRESHOLD !!!!!!!!!!!!!!!")


if __name__ == "__main__":
    users = [folder for folder in os.listdir('./dataset/')]

    with Pool() as pool:
        pool.map(process_subject, users[:])

# users = [folder for folder in os.listdir('./dataset/')]
# for subjectID in users:
#     #subjectID = f'AJA{i:04}'
#     path      = f'./dataset/{subjectID}/image/{subjectID}'
#     files     = [os.path.join(path,file) for file in os.listdir(path) if file.endswith('.tif')]
#     files.sort()

#     files     = files[1:]   #remove sagittal image this may remove non-sagittal frame but let's omit this

#     print(f'Subject ID: {subjectID}')
#     print(f'Number of files: {len(files)}')

#     imgs = []#np.zeros((len(files), 512, 512, 3)).astype(np.uint8)
#     for idx, file in enumerate(files):
#         img = cv2.imread(file)#, flags=cv2.IMREAD_UNCHANGED)
#         H,W,C = img.shape
#         if img.shape != (512,512,3):
#             print(idx)
#         else:
#             imgs.append(img)
#         # try:
#         #     #imgs[idx,256-H//2:256+H//2,256-W//2:256+W//2] = img
#         #     imgs.append(img)
#         # except:
#         #     print(idx)
        
#     print(f'Image shape: {np.shape(imgs)}')
#     imgs = np.array(imgs)
#     diff = calculate_frame_diff(imgs)
#     indices = find_indices_exceeding_threshold(diff, 15)
#     print(indices)
#     print()
    
#     if len(indices) == 0:
#         print("!!!!!!!!!!!!! REDUCE THRESHOLD !!!!!!!!!!!!!!!")