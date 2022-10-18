import numpy as np
from scipy import fftpack


def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant') #TODO
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

def convolution_pad(image,kernel):
    h,w=image.shape
    hk,wk = kernel.shape
    image_pad = wrap_boundary_liu(image,opt_fft_size((h+hk-1,w+wk-1)))
    F_B=psf2otf(kernel,image_pad.shape )
    F_IMG=np.fft.fft2(image_pad)
    restored = np.fft.ifft2(F_B*F_IMG).real
    return restored[:h,:w]


def color_convolution(image,kernel):
    H,W,C = image.shape
    color=np.zeros(image.shape)
    for i in range(0,C):
        color[:,:,i]=convolution_pad(image[:,:,i],kernel)
    return color


'''
Reducing boundary artifacts
'''


def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    #LUT_size = 2048
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
    return m


def wrap_boundary_liu(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H)-1
    k = np.arange(2, W)-1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k+1)] + boundary_image[np.ix_(j, k-1)] + boundary_image[np.ix_(j-1, k)] + boundary_image[np.ix_(j+1, k)]
    
    del(j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del(f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]
    del(f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0)/2
    else:
        tt = fftpack.dst(f2, type=1)/2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0)/2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1)/2) 
    del(f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)

    # divide
    f3 = f2sin/denom
    del(f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2, type=1, axis=1)/(2*(f3.shape[1]+1))
    else:
        tt = fftpack.idst(f3*2, type=1, axis=0)/(2*(f3.shape[0]+1))
    del(f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1)/(2*(tt.shape[0]+1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1, axis=0)/(2*(tt.shape[1]+1)))
    del(tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


