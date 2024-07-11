import cv2
vc = cv2.VideoCapture('/home/jian/CS/CS-MCNet/data/test/test.avi')
c=0
rval=vc.isOpened()
 
while rval:
  c = c + 1
  rval, frame = vc.read()
  if c == 20:
    break
  if rval:
    cv2.imwrite('/home/jian/CS/testimages/'+'test'+str(c) + '.jpg', frame) #命名方式
    print(c)
  else:
    break
vc.release()