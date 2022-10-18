import torch
import torch.fft
import numpy as np
from scipy.io import loadmat
import cv2
print(torch.__version__)

def dst_torch(y):
    '''
    input :(N,C,H,W)
    '''
    (N,C,H,W) = y.shape
    N_,C_,H_,W_=(N,C,H,W+1)
    y_pad = torch.zeros((N_,C_,H_,W_))
    y_pad[:,:,:,1:W_].copy_(y)
    y2 = torch.zeros(N_,C_,H_,W_*2)
    y2[:,:,:,0] = 0.0
    y2[:,:,:,W_] = 0.0
    y2[:,:,:,1:W_].copy_(y_pad[:,:,:,1:])
    
    y2[:,:,:,W_+1:].copy_(-y_pad[:,:,:,1:].flip(dims=[3]))

    #print(y2)
    a = -torch.fft.rfft(y2,dim=3)[:,:,:,:W_].imag
    return a[:,:,:,1:]

def idst_torch(y):
    '''
    input :(N,C,H,W)
    '''
    (N,C,H,W) = y.shape
    N_,C_,H_,W_=(N,C,H,W+1)
    y_pad = torch.zeros((N_,C_,H_,W_))
    y_pad[:,:,:,1:W_].copy_(y)
    c = torch.zeros(N_,C_,H_,W_+1,dtype=torch.cdouble) #TODO Hard code define double
    c[:,:,:,0] = 0.0
    c[:,:,:,W_] = 0.0
    c[:,:,:,1:W_].copy_(-1j*y_pad[:,:,:,1:])
    #print("pad before fft",c)
    c = torch.fft.irfft(c,dim=-1)[:,:,:,:W_]
    #print("depad after fft",c[:,:,:,:])#*(2*W_))
    return c[:,:,:,1:]*(2*W_)

def solve_min_laplacian_torch(boundary_image,device = "cuda"):
    #function [img_direct] = poisson_solver_function(gx,gy,boundary_image)
    #Inputs; Gx and Gy -> Gradients
    #Boundary Image -> Boundary image intensities
    #Gx Gy and boundary image should be of same size

    (N,C,H,W) = boundary_image.shape
    # Laplacian
    f = torch.zeros(N,C,H,W).to(device)
    # boundary image contains image intensities at boundaries
    boundary_image[:,:,1:-1, 1:-1] = 0;
    #print(boundary_image)
    j = torch.arange(2,H)-1;      
    k = torch.arange(2,W)-1;
    n = torch.arange(0,N); 
    c = torch.arange(0,C); 
    f_bp = torch.zeros((N,C,H,W)).to(device);
    n_idx,c_idx,j_idx,k_idx = torch.meshgrid(n,c,j,k)
    # TODO 
    f_bp[:,:,j_idx,k_idx] = -4*boundary_image[:,:,j_idx,k_idx]  + boundary_image[:,:,j_idx,k_idx+1] + boundary_image[:,:,j_idx,k_idx-1] + boundary_image[:,:,j_idx+1,k_idx] + boundary_image[:,:,j_idx-1,k_idx];

    f1 = f - f_bp; # subtract boundary points contribution
    # DST Sine Transform algo starts here
    f2 = f1[:,:,1:-1,1:-1];
    #print("f2,",f2)
    #del f1
    tt = dst_torch(f2)/2;
    #print("tt",tt)
    tt =dst_torch(tt.transpose(-1,-2))/2
    f2sin = torch.transpose(tt,-1,-2);  
    #del tt 
    #print("f2sin shape:",f2sin)
    # compute Eigen Values
    n_idx,c_idx,j_idx,k_idx = torch.meshgrid(n,c,torch.arange(1,H-1), torch.arange(1,W-1));
    denom = (2*torch.cos(np.pi*j_idx/(H-1))-2) + (2*torch.cos(np.pi*k_idx/(W-1)) - 2);
    #print("denom shape:",denom)
    # divide
    f3 = f2sin/denom;  
    #print("f3",f3)
    # compute Inverse Sine Transform
    tt = idst_torch(f3*2)/(2*(f3.shape[-1]+1));
    #print("idst",idst_torch(f3*2))
    tt = idst_torch(tt.transpose(-1,-2)*2)
    img_tt = torch.transpose(tt,-1,-2)
    #print("2 idst non scale",img_tt)
    img_tt = img_tt/(2*(img_tt.shape[-2]+1)); 
    #print("2 idst",img_tt)
    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image;
    img_direct[:,:,1:-1,1:-1] = 0;
    img_direct[:,:,1:-1,1:-1].copy_(img_tt);
    return img_direct 

def wrap_boundary_RGB_torch(img, img_size,device):
    """
    Reducing boundary artifacts in image deconvolution
    ICIP 2008
    input : (N,C,H,W)
    we also handle gray scale image
    """
    if img.shape[1] == 3:
        R = wrap_boundary_torch(img[:,0:1,:,:], img_size,device)
        G = wrap_boundary_torch(img[:,1:2,:,:], img_size,device)
        B = wrap_boundary_torch(img[:,2:3,:,:], img_size,device)
        RGB =  torch.cat((R,G,B),dim=1)
        return RGB
    elif  img.shape[1] == 1:
        return wrap_boundary_torch(img, img_size,device)
    else :
        print("input image fails")
        assert(False)
    
def wrap_boundary_torch(img, img_size,device):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    input : (N,C,H,W)
    """
    (N,C,H,W) = img.shape
    H_w = img_size[0] - H
    W_w = img_size[1] - W

    alpha = 1 
    HG = img[:,:,:,:]

    r_A = torch.zeros((N,C,alpha*2+H_w,W)).to(device);
    r_A[:,:,:alpha,:].copy_(HG[:,:,-alpha:,:]);
    r_A[:,:,-alpha:,:].copy_(HG[:,:,:alpha,:]);
    a = (torch.arange(H_w)/(H_w-1)).to(device);
    #r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1);
    r_A[:,:,alpha:-alpha,0].copy_( (1-a)*r_A[:,:,alpha-1,0] + a*r_A[:,:,-alpha,0]);
    #r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end);
    r_A[:,:,alpha:-alpha, -1].copy_( (1-a)*r_A[:,:,alpha-1,-1] + a*r_A[:,:,-alpha,-1]);

    r_B = torch.zeros((N,C,H, alpha*2+W_w)).to(device);
    r_B[:,:,:, :alpha].copy_( HG[:,:,:, -alpha:]);
    r_B[:,:,:, -alpha:].copy_(HG[:,:,:, :alpha]);
    a = (torch.arange(W_w)/(W_w-1)).to(device);
    #print((1-a).shape)
    #print(r_B[:,:,0,alpha-1].shape)
    #print((1-a)* r_B[:,:,0,alpha-1]+a*r_A[:,:,-alpha,0])
    #print(r_A[:,:,alpha:-alpha,0] )
    r_B[:,:,0, alpha:-alpha].copy_((1-a)*r_B[:,:,0,alpha-1] + a*r_B[:,:,0,-alpha]);
    r_B[:,:,-1, alpha:-alpha].copy_((1-a)*r_B[:,:,-1,alpha-1] + a*r_B[:,:,-1,-alpha]);

    # TODO

    if alpha == 1:
        #print(r_A)
        A2 = solve_min_laplacian_torch(r_A[:,:,alpha-1:,:],device=device);
        B2 = solve_min_laplacian_torch(r_B[:,:,:,alpha-1:],device=device);
        r_A[:,:,alpha-1:,:].copy_(A2);
        r_B[:,:,:,alpha-1:].copy_(B2);
    else:
        A2 = solve_min_laplacian_torch(r_A[:,:,alpha-1:-alpha+1,:],device=device);
        r_A[:,:,alpha-1:-alpha+1,:].copy_(A2);
        B2 = solve_min_laplacian_torch(r_B[:,:,:,alpha-1:-alpha+1],device=device);
        r_B[:,:,:,alpha-1:-alpha+1].copy_(B2);
    A = r_A;
    B = r_B;
    r_C = torch.zeros((N,C,alpha*2+H_w, alpha*2+W_w)).to(device);
    r_C[:,:,:alpha, :].copy_(B[:,:,-alpha:, :]);
    r_C[:,:,-alpha:, :].copy_(B[:,:,:alpha, :]);
    r_C[:,:,:, :alpha].copy_(A[:,:,:, -alpha:]);
    r_C[:,:,:, -alpha:].copy_(A[:,:,:, :alpha]);

    if alpha == 1:
        C2  = solve_min_laplacian_torch(r_C[:,:,alpha-1:, alpha-1:],device=device);
        r_C[:,:,alpha-1:, alpha-1:].copy_(C2);
    else:
        C2 = solve_min_laplacian_torch(r_C[:,:,alpha-1:-alpha+1, alpha-1:-alpha+1],device=device);
        r_C[:,:,alpha-1:-alpha+1, alpha-1:-alpha+1].copy_(C2);
    C = r_C;
    #return C;
    A = A[:,:,alpha-1:-alpha-1, :].clone();
    B = B[:,:,:, alpha:-alpha].clone();
    C = C[:,:,alpha:-alpha, alpha:-alpha].clone();
    #print(img.shape)
    #print(A.shape)
    #print(B.shape)
    #print(C.shape)
    #ret = torch.vstack((torch.hstack((img,B)),torch.hstack((A,C))));
    ret = torch.cat((torch.cat((img,B),-1),torch.cat((A,C),-1)),-2);
    return ret;

def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 8192
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = np.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7-1] = e7
                    if e7*11 <= LUT_size:
                        opt_fft_size_LUT[e7*11-1] = e7*11
                    if e7*13 <= LUT_size:
                        opt_fft_size_LUT[e7*13-1] = e7*13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i-1] != 0:
            nn = i-1
        else:
            opt_fft_size_LUT[i-1] = nn+1

    m = np.zeros(len(n))
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m[c] = opt_fft_size_LUT[nn-1]
        else:
            m[c] = -1
    return [int(x) for x in m]
def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

def convolution(image,kernel):
    h,w=image.shape
    #F_B=np.fft.fft2(kernel,(h,w) )
    F_B=psf2otf(kernel,(h,w) )
    F_IMG=np.fft.fft2(image)
    restored = np.fft.ifft2(F_B*F_IMG).real
    return restored
if __name__ == "__main__":
    np.random.seed(123)
    #img_test =np.random.random((3,5))
    #img_test = torch.FloatTensor(img_test).reshape(1,1,3,5)
    ##img_test =torch.ones(1,1,7,5)
    ##print(idst_torch(img_test))
    #print(wrap_boundary_torch(img_test,(10,10)))
    ##print(solve_min_laplacian_torch(img_test))
    MAT_PATH="/home/r09021/unfolding_research/Motion_deblur/Benchmark/2009Levin/Levin09blurdata/im05_flit01.mat"
    #print(MAT_PATH)
    # synthetic data
    img_mat = loadmat(MAT_PATH)
    img_sharp=img_mat["x"]
    img_kernel = img_mat["f"]
    img_kernel =img_kernel/np.sum(img_kernel)
    img_blurred=convolution(img_sharp,img_kernel) # synthetic blurred image
    # blurred shape
    H,W=img_blurred.shape
    shape =(H + img_kernel.shape[0]-1,W + img_kernel.shape[1]-1 )
    print(img_kernel.shape)
    #print(opt_fft_size(shape))
    img_blurred = torch.FloatTensor(img_blurred).unsqueeze(0).unsqueeze(0)
    pad_blurred = np.clip(wrap_boundary_torch(img_blurred,opt_fft_size(shape),"cuda:0")*255.0,0,255,)
    cv2.imwrite("blurred_pad_torch.png",pad_blurred.numpy().squeeze(0).squeeze(0))