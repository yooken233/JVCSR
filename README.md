
Test usage: test video path in cofing.py .

> python test.py test --height=1080 --width=1920 --load_model_path='./checkpoints/cr8_120_0.01/best_model.pth' --frame_num=5 --cr=0.125 --weights='./weights/weights_cr8.txt'

default test result is in ./results and other parameters can be find in config.py.

Train usage: download our train dataset and put it in ./data/train. 
           
> python train.py train --save_train_root='./checkpoints/default_train' --cr=0.125 --weights='./weights/weights_cr16.txt' --batch_size=4 --max_epoch=50 --print_freq=10 --lr=0.01

default train result is in cofing.py 


