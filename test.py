from config import opt 
import torch as t 
import torch.nn as nn
import numpy as np 
import os 
from models.MCNet import MCNet
#from models.cbam_before_fb import SRFBN
from models.SkipRR_arch import SKIPRR
from models.DnCNN_Refine import SRFBN
import utils 
import math 
from skimage.measure import compare_psnr 
from skimage.measure import compare_ssim
from scipy.ndimage.interpolation import zoom
import time
import torchsnooper
import cv2
from solvers import create_solver
import options.options as option
from data import common
from skimage.metrics import peak_signal_noise_ratio as psnrf
from skimage.metrics import structural_similarity as ssimf


############################.113

def _overlap_crop_forward(x, shave=10, min_size=100000, bic=None):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 2
    scale = 2
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if bic is not None:
        bic_h_size = h_size*scale
        bic_w_size = w_size*scale
        bic_h = h*scale
        bic_w = w*scale
        
        bic_list = [
            bic[:, :, 0:bic_h_size, 0:bic_w_size],
            bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
            bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
            bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if bic is not None:
                bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

            sr_batch_temp = model(lr_batch)

            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp

            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            _overlap_crop_forward(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
            ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
    
def testDenoise(LR):
    with torch.no_grad():
        forward_func = _overlap_crop_forward
        SR = forward_func(LR)
        if isinstance(SR, list):
            SR = SR[-1]
        else:
            SR = SR
    return SR


@t.no_grad()
def test(**kwargs):
    #refresh parameter
    opt._parse(kwargs)
    
    # Image Enhancement
    optdenoise = option.parse(opt.denoiseopt)
    #print(optdenoise)
    optdenoise = option.dict_to_nonedict(optdenoise)
    scale = optdenoise['scale']
    degrad = optdenoise['degradation']
    network_opt = optdenoise['networks']
    model_name = network_opt['which_model'].upper()
    solver = create_solver(optdenoise)
    
    #create save folder, open log file
    save_test_root = opt.save_test_root
    if not os.path.exists(save_test_root):
        os.makedirs(save_test_root)
    log_file = open(save_test_root+"/result.txt",mode='w')
    
    #set gpu environment, load weights
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_visible_devices)
    bernoulli_weights = np.loadtxt(opt.weights)
    bernoulli_weights = t.from_numpy(bernoulli_weights).float().to(opt.device)

    model = MCNet(bernoulli_weights,opt.cr,opt.blk_size,opt.ref_size).eval()
    
    # upsamplingmodel = SRFBN()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device) 
    #denoisemodel.to(opt.device) 
    
    model.eval()

    #get test videos
    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    print("total test video number:",video_num)

    end = time.time()
    psnr_av = 0
    ssim_av = 0
    time_av = 0
    
    # [yang] Get the original video of reshape and RGBtoGray(1 channel)
    # for item in videos:
        # if (item.split(".")[-1]!='mp4'):
                # continue
        # uv = utils.Video(opt.height,opt.width)
        # test_data_or = uv.video2array(item,opt.frame_num)
        # test_data_or_t = t.from_numpy(test_data_or).float().to(opt.device)
        
        # uv.array2video(test_data_or_t/255,opt.save_test_or_root,opt.save_or_root_txt)
        
    for item in videos:
        if (item.split(".")[-1]!='mp4'):
            continue
        print("now is processing:",item)
        log_file.write("%s"%item)
        log_file.write("\n")
        #print("opt.height,opt.width",opt.height,opt.width)
        uv = utils.Video(opt.height,opt.width)
        # vc = cv2.VideoCapture(item)
        # frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))       
        # frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # if ((frame_width != opt.width)|(frame_height != opt.height)):
            # print("size mismatch, skip this video!")
            # continue
        test_data = uv.video2array(item,opt.frame_num)
        #test_data_or = uv.video2array(item,opt.frame_num)
        test_data_t = t.from_numpy(test_data).float().to(opt.device)
        
        
        #uv.array2video(test_data_t/255,opt.save_down_root,opt.save_down_root_txt)
        
        result_data_t = t.zeros_like(test_data_t).cuda()

        psnr_total = 0
        ssim_total = 0
        frame_cnt = 0
    
        
        
        #do test on every video
        for i in range(test_data_t.size(0)): # test_data_t.size(0)= blk_num_h  
            for j in range(test_data_t.size(1)):# test_data_t.size(1)= blk_num_w 
                frames = test_data_t[i,j,:,:,:]
                #print("frames",frames.shape)
                frames_num = frames.size(0)
        
                result_frame = t.ones(1,frames[0].size(0),frames[0].size(1)).float().to(opt.device)
                result_frames = t.zeros(frames_num,frames[0].size(0),frames[0].size(1)).to(opt.device)
                frames_t = frames
                
                #or_frame = t.ones(1,frames[0].size(0),frames[0].size(1)).float().to(opt.device)
                    
                x_b = uv.frame_unfold(frames_t,opt.blk_size,int(opt.blk_size/2)).to(opt.device)
                #print("x_b",x_b.shape)
                blk_num_h = x_b.size(1)
                blk_num_w = x_b.size(2)

                for ii in range(frames_num):
                    x_ref_b = uv.frame_unfold(result_frame,opt.ref_size,int(opt.ref_size/2))
                   
                    #frames_one = frames[ii,:,:].unsqueeze(0) 
                    #x_ref_b = uv.frame_unfold(frames_one/255.0,opt.ref_size,int(opt.ref_size/2))
                   
                    result_b = t.zeros_like(x_b[0].unsqueeze_(0))
                   
                    input_ = (x_b[ii,:,:,:,:]/255.0).float().to(opt.device)
                
                    input_target = input_.view(1*blk_num_h*blk_num_w,opt.blk_size,opt.blk_size)
                    input_m = input_.view(1*blk_num_h*blk_num_w,opt.blk_size*opt.blk_size,1)

                    ref = x_ref_b.repeat(1,2,1,1,1)
                    ref[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],:,:,:] = ref[:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17],:,:,:]
                    ref = ref.repeat(1,1,2,1,1)
                    ref[:,:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],:,:] = ref[:,:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17],:,:]
                    ref_cat = ref[:,:,-1,:,:].view(1,ref.size(1),1,opt.ref_size,opt.ref_size)
                    ref = t.cat((ref,ref_cat),2)
                    ref_cat = ref[:,-1,:,:,:].view(1,1,ref.size(2),opt.ref_size,opt.ref_size)
                    ref = t.cat((ref,ref_cat),1)
                    ref = ref.view(1*blk_num_h*blk_num_w,opt.ref_size,opt.ref_size)

                    

                    b_s = input_target.size(0)
                    weight = bernoulli_weights.unsqueeze(0).repeat(b_s,1,1).to(opt.device)
                    input = t.bmm(weight,input_m).squeeze_(2)
                    if(opt.noise_snr>0):
                        input = add_noise(input,opt.noise_snr,10)
                        
                        
                    output,_ = model(input,ref,input)
                    #print("output:",output.shape)
                    result_b = output.view(1,blk_num_h,blk_num_w,opt.blk_size,opt.blk_size)
                    #print(result_b.shape)
                    frame_cnt = frame_cnt + 1
                    result_frame = uv.frame_fold(result_b,opt.blk_size,int(opt.blk_size/2))
                    #print("result_frame outputed from Huang's",result_frame)
                    #print("result_frame.shape outputed from Huang's",result_frame.shape)
                    #print("result_frame", result_frame)
                    
                    #our denoise part
                    # result_frame = result_frame[0,:,:].cpu()
                    # result_frame = np.expand_dims(result_frame, axis=2)
                    # result_frame = common.np2Tensor([result_frame], optdenoise['rgb_range'])[0]
                    # result_frame = t.unsqueeze(result_frame,0).to(opt.device)
                    # solver.test1image(result_frame)
                    # result_frame = t.squeeze(result_frame,0).to(opt.device)                     
                    
                    
                
                    result_frames[ii] = result_frame
                    
                    
                    #psnr = compare_psnr(frames_t[ii].cpu().numpy().astype(dtype=np.double),(result_frame*255).squeeze(0).cpu().numpy().astype(dtype=np.double),data_range=255.0)
                    #psnr = psnrf(frames_t[ii].cpu().numpy().astype(dtype=np.double),(result_frame*255).squeeze(0).cpu().numpy().astype(dtype=np.double),data_range=255.0)
                    psnr = psnrf(frames_t[ii].cpu().numpy(),(result_frame*255).squeeze(0).cpu().numpy(),data_range=255.0)
                    ssim = ssimf(frames_t[ii].cpu().numpy(),(result_frame*255).squeeze(0).cpu().numpy())
                    #print("psnr, ssim: ",psnr,ssim)
                    psnr_total = psnr_total + psnr
                    ssim_total = ssim_total + ssim
                    
                    
                    
                result_data_t[i,j,:,:,:] = result_frames
                
                
        #
        
        # whole_video = uv.array2(result_data_t)
        # for jj in range(len(whole_video)):
            # result_frame = whole_video[0,:,:]
            # result_frame = np.expand_dims(result_frame, axis=2)
            # result_frame = common.np2Tensor([result_frame], optdenoise['rgb_range'])[0]
            # result_frame = t.unsqueeze(result_frame,0).to(opt.device)
            # solver.test1image(result_frame)
            # result_frame = t.squeeze(result_frame,0).to(opt.device)
        # uv.writevideo(result_frame*255,opt.save_whole_root)
         
               
               
        
        #uv.array2(result_data_t,opt.save_test_root)
        uv.array2video(result_data_t,opt.save_test_root,opt.save_result_root_txt)
       
        
        
        #get log information
        video_time = time.time() - end
        info = str(psnr_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(ssim_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(video_time/frame_cnt)+"\n"
        log_file.write("%s"%info)
        end = time.time()
        
        print("PSNR is:",psnr_total/frame_cnt,"SSIM is:",ssim_total/frame_cnt,"Time per frame is:",video_time/frame_cnt)
        psnr_av = psnr_av + psnr_total/frame_cnt
        ssim_av = ssim_av + ssim_total/frame_cnt
        time_av = time_av + video_time/frame_cnt
    
    print("******************************************************************************")
    print("Average PSNR is:",psnr_av/video_num,"Average SSIM is:",ssim_av/video_num,"Average Time per frame is:",time_av/video_num)
    info ="\n"+"Average PSNR and SSIM"+"\n"
    log_file.write("%s"%info)
    info = str(psnr_av/video_num)+"\n"
    log_file.write("%s"%info)
    info = str(ssim_av/video_num)+"\n"
    log_file.write("%s"%info)
    
    info ="\n"+str(opt.load_model_path)
    log_file.write("%s"%info)
    
    log_file.close()
def add_noise(input,SNR,seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True

    input_np = input.cpu().numpy()
    noise = np.random.randn(input_np.shape[0],input_np.shape[1]).astype(np.float32)
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(input_np)**2/(input_np.size)
    noise_power = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_power)/np.std(noise))*noise

    y = input_np + noise 
    y = t.from_numpy(y).cuda()
    return y

if __name__=='__main__':
    import fire
    fire.Fire()






            