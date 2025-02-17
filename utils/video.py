import cv2 
import numpy as np 
import os 
import imageio 
from torchvision import transforms as tf 
from scipy.ndimage.interpolation import zoom
import torch as t 
import fire
import torchsnooper


class Video(object):
    def __init__(self,height,width):
        self.name = None
        self.height = height
        self.width = width
        self.blk_num_h = None
        self.blk_num_w = None
        self.input_patch_numel = None
        self.input_patch_size = None
        self.input_frame_numel = None 
        self.input_frame_size = None 
    
    def frame_unfold(self,frame_imgs_t,block,n=0):
        #input data type torch tensor
        #input data shape [frame_num,height,width]
        #output data type torch tensor
        #output data size [frame_num,block_num_h,block_num_w,block_width,block_width]
        
        self.input_frame_size = frame_imgs_t.size()
        self.input_frame_numel = frame_imgs_t.numel()#multiple of all numbers in the matrix
        self.frame_h = frame_imgs_t.size(1)
        #print("frame_imgs_t",frame_imgs_t.shape)
        self.frame_w = frame_imgs_t.size(2)
        if((self.frame_h-block)%(block-n)!=0 or (self.frame_w-block)%(block-n)!=0):
            print("frame size is",self.input_frame_size,"error:size mismatch!")
            return 0
        else:
            output_ = frame_imgs_t.unfold(1,block,block-n).unfold(2,block,block-n)
            #print("output_.shape:", output_.shape)
            output = output_.contiguous()
            self.input_patch_numel = output.numel()
            self.input_patch_size = output.size()
            return output

    #@torchsnooper.snoop()
    def frame_fold(self,input_patches,block,n=0):
        #input data type torch tensor
        #input data shape [frame_num,block_num_h,block_num_w,block_width,block_width]
        #output data type torch tensor
        #output data shape [frame_num,height,width]
        input_patches = input_patches.float()
        #print("input_patches: ",input_patches.shape)
        idx = t.zeros(self.input_frame_numel).long().to('cuda')
        t.arange(0,self.input_frame_numel,out=idx)
        idx = idx.view(self.input_frame_size)
        idx_unfold = idx.unfold(1,block,block-n).unfold(2,block,block-n)
        idx_unfold = idx_unfold.contiguous().view(-1)
        
        video = t.zeros(self.input_frame_size).view(-1).to('cuda')
        #print("video",video.shape)
        video_ones = t.zeros(self.input_frame_size).view(-1).to('cuda')
        #print("video_ones",video_ones.shape)
        patches_ones = (t.zeros_like(input_patches)+1).view(-1).to('cuda')
        #print("patches_ones",patches_ones.shape)
        input_patches = input_patches.contiguous().view(-1)
        video.index_add_(0,idx_unfold,input_patches)
        #print("video",video.shape)
        video_ones.index_add_(0,idx_unfold,patches_ones)
        #print("video_ones",video_ones.shape)
        output = (video/video_ones).view(self.input_frame_size)
        #print("output",output.shape)
        return output
     
    def video2array(self,filename,frame_num):
        self.name = filename.split(".")[-2].split("/")[-1]
        self.blk_num_w = int(self.width/160)
        self.blk_num_h = int(self.height/160)
        blk_frames = np.zeros((self.blk_num_h,self.blk_num_w,frame_num,160,160))

        vc = cv2.VideoCapture(filename)
        frame_num_cv2=vc.get(7)
        frame_index = 0
        
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False

        if frame_num > frame_num_cv2:
            frame_num = frame_num_cv2-1

        while (rval and frame_index<frame_num):
            rval, frame = vc.read()
            frame_yt = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            for j in range(self.blk_num_h):
                for k in range(self.blk_num_w):
                    blk_frames[j,k,frame_index,:,:] = frame_yt[j*160:(j+1)*160,k*160:(k+1)*160]
            frame_index = frame_index + 1
            cv2.waitKey(1)
        vc.release()

        print("job done!")
        return blk_frames
        
    def video2array_down(self,filename,frame_num):
        self.name = filename.split(".")[-2].split("/")[-1]
        # [yang] divided by 2 because of downsampling
        self.blk_num_w = int(self.width/160/2)
        self.blk_num_h = int(self.height/160/2)
        blk_frames = np.zeros((self.blk_num_h,self.blk_num_w,frame_num,160,160))

        vc = cv2.VideoCapture(filename)
        frame_num_cv2=vc.get(7)
        frame_index = 0
        
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False

        if frame_num > frame_num_cv2:
            frame_num = frame_num_cv2-1

        while (rval and frame_index<frame_num):
            rval, frame = vc.read()
            frame_yt = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #[yang] downsampling here
            
            frame_yt = cv2.resize(frame_yt,(0, 0),fx=0.5, fy=0.5,interpolation=cv2.INTER_CUBIC)
            # savename = savetxtroot + "/" + self.name+ str(frame_index) + ".txt"
            # np.savetxt(savename,frame_yt[frame_index].astype(np.uint8),fmt='%d', delimiter="\t")
            
            for j in range(self.blk_num_h):
                for k in range(self.blk_num_w):
                    blk_frames[j,k,frame_index,:,:] = frame_yt[j*160:(j+1)*160,k*160:(k+1)*160]
                    
            frame_index = frame_index + 1
            cv2.waitKey(1)
        vc.release()

        print("job done!")
        return blk_frames
        

    
       
    #@torchsnooper.snoop()
    def array2video(self,data,saveroot,savetxtroot):
        data = data.cpu()
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        frames_num = data.size(2)
        frames = t.zeros((frames_num,self.blk_num_h*160,self.blk_num_w*160))
        frames_np = np.zeros((frames_num,self.blk_num_h*160,self.blk_num_w*160))
        for i in range(self.blk_num_h):
            for j in range(self.blk_num_w):
                frames[:,i*160:(i+1)*160,j*160:(j+1)*160] = data[i,j,:,:,:]

        for i in range(frames.size(0)):
            frames_t = tf.ToPILImage()(frames[i])
            frames_np[i] = np.asarray(frames_t)
            savename = savetxtroot + "/" + self.name+str(i) + ".txt"
            np.savetxt(savename,frames_np[i].astype(np.uint8),fmt='%d', delimiter="\t")
        frames_np = frames_np.astype(np.uint8)
        
        savename = saveroot + "/" + self.name + ".avi"
        
        print("save video path is:",savename)
        
        imageio.mimwrite(savename,frames_np,format='avi',fps=30)
        return None
        
    def array2video_2080(self,data,saveroot,savetxtroot):
        data = data.cpu()
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        frames_num = data.size(2)
        frames = t.zeros((frames_num,(self.blk_num_h-1)*160,self.blk_num_w*160))
        frames_np = np.zeros((frames_num,(self.blk_num_h-1)*160,self.blk_num_w*160))
        for i in range(self.blk_num_h-1):
            for j in range(self.blk_num_w):
                frames[:,i*160:(i+1)*160,j*160:(j+1)*160] = data[i,j,:,:,:]

        for i in range(frames.size(0)):
            frames_t = tf.ToPILImage()(frames[i])
            frames_np[i] = np.asarray(frames_t)
            savename = savetxtroot + "/" + self.name+str(i) + ".txt"
            np.savetxt(savename,frames_np[i].astype(np.uint8),fmt='%d', delimiter="\t")
        frames_np = frames_np.astype(np.uint8)
        
        savename = saveroot + "/" + self.name + ".avi"
        
        print("save video path is:",savename)
        
        imageio.mimwrite(savename,frames_np,format='avi',fps=30)
        return None
        
    def array2(self,data):
        data = data.cpu()
      
        frames_num = data.size(2)
        frames = t.zeros((frames_num,self.blk_num_h*160,self.blk_num_w*160))
        frames_np = np.zeros((frames_num,self.blk_num_h*160,self.blk_num_w*160))
        for i in range(self.blk_num_h):
            for j in range(self.blk_num_w):
                frames[:,i*160:(i+1)*160,j*160:(j+1)*160] = data[i,j,:,:,:]

        for i in range(frames.size(0)):
            frames_t = tf.ToPILImage()(frames[i])
            frames_np[i] = np.asarray(frames_t)
        frames_np = frames_np.astype(np.uint8)
    
        return frames_np

    def writevideo(self,data,saveroot):
        data = data.cpu()
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)
        frames_num = data.size(0)
        frames_np = np.zeros((frames_num,self.blk_num_h*160,self.blk_num_w*160))
        for i in range(frames_num):
            frames_t = tf.ToPILImage()(data[i])
            frames_np[i] = np.asarray(frames_t)
            
        frames_np = frames_np.astype(np.uint8)
        
        savename = saveroot + "/" + self.name + ".avi"
        
        print("save video path is:",savename)
        
        imageio.mimwrite(savename,frames_np,format='avi',fps=30)
        return None
    