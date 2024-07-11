import h5py
from PIL import Image
from operator import attrgetter
from torchvision import transforms as tf 
import os 
import torch as t
import numpy as np
import cv2

def frame_capture(loadpath,time_freq):
    #output data type list(numpy array)
    #output data shape(frame_num height width)
    img_idx = 1
    videopath = loadpath
    frame_imgs = []
    output = []
    print("video loadpath is",videopath)

    vc = cv2.VideoCapture(videopath)

    video_frame_idx = 1

    if vc.isOpened():
        rval,frame = vc.read()
    else:
        rval = False
            
    while rval:
        rval,frame = vc.read()
        if(video_frame_idx%time_freq == 0):
            frame_imgs.append(frame)
            img_idx = img_idx + 1
        video_frame_idx = video_frame_idx + 1
        cv2.waitKey(1)
    vc.release()
    frame_imgs.pop()
    for frame_item in frame_imgs:
        frame_item_ = cv2.cvtColor(frame_item,cv2.COLOR_BGR2GRAY)
        f_h,f_w = frame_item_.shape
        frame_item_ = frame_item_[int(f_h/2)-80:int(f_h/2)+80,int(f_w/2)-80:int(f_w/2)+80]
        output.append(frame_item_)
    return output

def frame_unfold(frame_imgs_t,block,overlap=0):
    #input data type torch tensor
    #input data shape [frame_num,height,width]
    #output data type torch tensor
    #output data size [frame_num,block_num_h,block_num_w,block_width,block_width]
    input_frame_size = frame_imgs_t.size()
    frame_h = frame_imgs_t.size(1)
    frame_w = frame_imgs_t.size(2)
    if((frame_h-block)%(block-overlap)!=0 or (frame_w-block)%(block-overlap)!=0):
        print("frame size is",input_frame_size,"error:overlap size mismatch!")
        return 0
    else:
        output_ = frame_imgs_t.unfold(1,block,block-overlap).unfold(2,block,block-overlap)
        output = output_.contiguous()
        return output

class filename(object):
    def __init__(self,name,g,c,f):
        self.name = name
        self.g = g
        self.c = c
        self.f = f
    def __repr(self):
        return repr((self.name,self.g,self.c,self.f))
    def get_filename(self):
        return self.name+'_g'+self.g+'_c'+self.c+'_'+self.f

class MCData(object):
    def __init__(self,root):
        self.root = root
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        imgs_num = len(imgs)
        train_file = []
        for item in imgs:
            name = item.split('.')[-2].split('_')[-4].split('/')[-1]
            g = item.split('.')[-2].split('_')[-3].split('g')[-1]
            c = item.split('.')[-2].split('_')[-2].split('c')[-1]
            f = item.split('.')[-2].split('_')[-1]
            filename_ = filename(name,g,c,f)
            train_file.append(filename_)
        train_file.sort(key=attrgetter('name','g','c','f'))
        train_dataset_ = train_file

        one_video_train_dataset = []
        self.train_dataset = []
        f_name = train_dataset_[0].name
        f_g = train_dataset_[0].g
        f_c = train_dataset_[0].c
        for item in train_dataset_:
            if(item.name==f_name and item.g==f_g and item.c==f_c):
                one_video_train_dataset.append(item)
            else:
                one_video_train_dataset.sort(key=lambda x:int(x.f))
                tmp_num = len(one_video_train_dataset)
                for item_1 in [one_video_train_dataset[i:i+2] for i in range(0,tmp_num,2)]:
                    if(len(item_1)==2):
                        self.train_dataset.append(item_1)
                    else:
                        break
                one_video_train_dataset = []
                one_video_train_dataset.append(item)
            f_name = item.name
            f_g = item.g
            f_c = item.c
        one_video_train_dataset.sort(key=lambda x:int(x.f))
        tmp_num = len(one_video_train_dataset)
        for item_1 in [one_video_train_dataset[i:i+2] for i in range(0,tmp_num,2)]:
            if(len(item_1)==2):
                self.train_dataset.append(item_1)
            else:
                break

        self.transforms = tf.Compose([tf.ToTensor()])
    
    def __getitem__(self,index):
        output = []
        img_path_object = self.train_dataset[index]
        for i in range(2):
            img_path = self.root+'/'+img_path_object[i].get_filename()+'.jpg'
            data = Image.open(img_path)
            data = self.transforms(data)
            output.append(data)
        return output
    
    def __len__(self):
        return len(self.train_dataset)

root = '/data2/jian/UCF-101/'
# videos = [os.path.join(root,video) for video in os.listdir(root)]
# videos_num = len(videos)
# print(videos_num)
folders = [os.path.join(root,folder) for folder in os.listdir(root)]
folders_num = len(folders)
print(folders_num)

for item in os.listdir(root):
    #item在此处已经是子目录的名字
   folder = os.path.join(root,item)
   videos = [os.path.join(folder,video) for video in os.listdir(folder)]
   print(len(videos))
   for item in videos:
       if(item.split('.')[-1]!='avi'):
           continue
       print(item)
       saveroot = "/data2/jian/CS-MCNet/img_2/"+item.split(".")[-2].split("/")[-1].split("_")[1]+"_"+item.split(".")[-2].split("/")[-1].split("_")[2]+"_"+item.split(".")[-2].split("/")[-1].split("_")[3]
           
       a = frame_capture(item,1)
       frame_cnt = 0
       for i in range(len(a)):
           savename_p2 = str(frame_cnt)+'.jpg'
           savepath = saveroot+'_'+savename_p2
           #print(savepath)
           img = a[frame_cnt]
           cv2.imwrite(savepath,img)
           frame_cnt = frame_cnt + 1

mcdata = MCData('/data2/jian/CS-MCNet/img_2')
data_num = mcdata.__len__()
print(data_num)
for i in range(data_num):
    train_data = mcdata.__getitem__(i)
    x_ = train_data[1].to('cuda')
    x_ref_ = train_data[0].to('cuda')
    x_b = frame_unfold(x_,16,0)
    x_ref_b = frame_unfold(x_ref_,32,0)
    ref = x_ref_b.repeat(1,2,1,1,1)
    ref[:,[0,1,2,3,4,5,6,7,8,9],:,:,:] = ref[:,[0,5,1,6,2,7,3,8,4,9],:,:,:]
    ref = ref.repeat(1,1,2,1,1)
    ref[:,:,[0,1,2,3,4,5,6,7,8,9],:,:] = ref[:,:,[0,5,1,6,2,7,3,8,4,9],:,:]
    file_name = '/data2/jian/CS-MCNet/h5_2/MCData_' + str(i) + '.h5'
    with h5py.File(file_name,'w') as f:
        f.create_dataset('imgs_data',data = x_b.cpu().numpy())
        f.create_dataset('ref_data',data=ref.cpu().numpy())
    if(i%1000==0):
        print("have done",i)
