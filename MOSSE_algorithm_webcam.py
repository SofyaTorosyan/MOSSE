# MOSSE tracking algorithm 
import argparse
import os 
import cv2
import numpy as np

# collects all images from dataset into some list
def get_img_list(img_path):

    # collect all image paths in list
    img_list = []
    for frame in os.listdir(img_path):
        if os.path.splitext(frame)[1] == '.jpg':
            img_list.append(os.path.join(img_path, frame))

    # sort img_paths, so seuqence of frames will be right.        
    img_list.sort()
    return img_list


# used for linear mapping...
def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

# get the ground-truth gaussian reponse
# syntetically generate desired correlation output
def get_gauss_response(args, gray_img, init_gt):

    # get the shape of the image
    height, width = gray_img.shape

    # return rectangular grid out of two given one-dimensional arrays 
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # get the center of the object
    center_x = init_gt[0] + 0.5 * init_gt[2] # move x coordinate of ROI
    center_y = init_gt[1] + 0.5 * init_gt[3] # move y coordinate of ROI

    # calculate the distance, dist is matrix
    dist = (np.square(x - center_x) + np.square(y - center_y)) / (2 * args.sigma)
   
    # get the response map
    response = np.exp(-dist)

    # normalize, the same shape as dist 
    response = linear_mapping(response)
    
    return response

# FFT assumes that signal repeats forever. But it needs only part of the signal.
def window_func_2d(height, width):
    win_col = np.hanning(width) # domain is from 0-width
    win_row = np.hanning(height) # domain is from 0-width
    col, row = np.meshgrid(win_col, win_row) # return rectangular grid out of two given one-dimensional arrays 

    window = col * row

    return window

# pre-processing the image
def pre_process(img):
    # get the size of the img
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window # this is tracking window(selected with bounding box) of the image
    return img

def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image randomly
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    cv2.imshow("image", matrix_rot)
    cv2.waitKey(0)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

# pre train the filter on the first frame
# MOSSE filter:
# G is synthetic target on the first frame in the Fourier domain.
# np.fft.fft2(fi) is preprocessed cropped template  in fourier domain
# np.conjugate(np.fft.fft2(fi)) complex conjugate of preprocessed cropped template in fourier domain.
def pre_training(args, init_frame, G):
    height, width = G.shape
    fi = cv2.resize(init_frame, (width, height))
    # pre-process img
    fi = pre_process(fi)
    Ai = G * np.conjugate(np.fft.fft2(fi))
    Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))

    for _ in range(args.num_pretrain):
        if args.rotate:
            fi = pre_process(random_warp(init_frame))
        else:
            fi = pre_process(init_frame)
        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))  # sum all 
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) # sum all
    
    return Ai, Bi # return sumed values.


def MOSSE_tracking(args, img_path):

    cap = cv2.VideoCapture(0)
    # take first frame
    #img_list   = get_img_list(img_path)  # get list of images
    success, init_img = cap.read()
    init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY) # get gray scale image of the initial image
    init_frame = init_frame.astype(np.float32)

    # get the init ground truth [x, y, width, height]
    fromCenter = False
    init_gt = cv2.selectROI('demo', init_img, fromCenter) # region of interest/ground truth
    init_gt = np.array(init_gt).astype(np.int64) # 1d array containing upper left coordinates and width, height (4, )
    
    # start to draw the gaussian response
    # using image and ground truth, gaussian response is created, which has gaussian shape, 
    # and peak is on the center of the target(which should be tracked)
    response = get_gauss_response(args, init_frame, init_gt)

    # create the training set
    g  = response[init_gt[1] : init_gt[1] + init_gt[3], init_gt[0] : init_gt[0] + init_gt[2]]   # response[y: y + h, x: x + w]
    fi = init_frame[init_gt[1] : init_gt[1] + init_gt[3], init_gt[0] : init_gt[0] + init_gt[2]] # frame[y: y + h, x: x + w]
    G = np.fft.fft2(g) # convert syntethic target image to the fourier domain using FFT 

    # pre-train filter on the first frame
    Ai, Bi = pre_training(args, fi, G)
    Ai = args.lr * Ai
    Bi = args.lr * Bi
    pos = init_gt.copy()
    clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64) # get the position

    # start the tracking
    while True:
        # preprocess current frame, change to gray scale.
        success, current_frame = cap.read()
        if(success == False):
            break
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float32)
           
        # else if this isn't the first frame
        Hi = Ai / Bi # update the filter
        # tracking window is cropped from the frame with the tracking window centered on the position of the object from the previous frame
        fi = frame_gray[clip_pos[1]: clip_pos[3], clip_pos[0]: clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3]))) # tracking window is preprocessed
        # this results in the peak, which indicates new position of the object           
        Gi = Hi * np.fft.fft2(fi) # tracking window is multiplied with the filter to get new position of the object in the current frame
        gi = linear_mapping(np.fft.ifft2(Gi))
        # find the max position
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
            
        # update the position...
        pos[0] = pos[0] + dx
        pos[1] = pos[1] + dy
        # trying to get the clipped position [xmin, ymin, xmax, ymax]
        clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
        clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
        clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
        clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
        clip_pos = clip_pos.astype(np.int64)
        # get the current frame
        fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
        # online update
        Ai = args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - args.lr) * Ai
        Bi = args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - args.lr) * Bi
            
        # visualize the tracking process
        cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
        cv2.imshow('demo', current_frame)
        cv2.waitKey(100)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--lr',           type = float,          default = 0.1, help = 'the learning rate')
    parse.add_argument('--sigma',        type = float,          default = 2,   help = 'the sigma')
    parse.add_argument('--num_pretrain', type = int,            default = 128,   help = 'the number of pretrain')
    parse.add_argument('--rotate',       action = 'store_true',                  help = 'if rotate image during pre-training.')
    parse.add_argument('--record',       action = 'store_true',                  help = 'record the frames')
    
    args = parse.parse_args()
    img_path = 'datasets/surfer/'

    MOSSE_tracking(args, img_path)