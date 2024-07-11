
import cv2
import glob
import imageio
import os


    
def images_to_video(path):
    img_array = []
    

    for filename in glob.glob(path+'/*.b'):
        img = cv2.imread(filename)
        print(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)
 
    # 图片的大小需要一致
    #img_array, size = resize(img_array, 'largest')
    fps = 30
    out = cv2.VideoWriter('/home1/jian/JVCSR/SR_Output/ours/3.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,640))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
def main():
    path = "/home1/jian/ForV7/ISTA/sr01/1/"
    images_to_video(path)
 
if __name__ == "__main__":
    main()