import numpy as np
from scipy.sparse.linalg import svds

def norm_img(img):
    min_i = img.min()
    max_i = img.max()
    img = (img - min_i)/(max_i - min_i)
    return img

def to_8bit(img): 
    return (255 * img).astype(np.uint8)

def gamma_correct(img, gamma=1/2.2):
    img = np.power(img, gamma)
    return img

def compute_base(L: np.ndarray, C: np.ndarray, R: np.ndarray, option: str):
    """
    Generate basis for reflectance and illumination.
    
    Args:
        L: Measured illuminant spectra data (N x M)
        C: Camera response spectra (N x 3)
        R: Measured reflectance spectra data (N x M)
        option: Method for generating basis ('joint', 'pca', 'wpca', or 'wpca1')
    
    Returns:
        UR: Basis for reflectance
        UL: Basis for illumination
    """
    if option == 'joint':
        VR, _, VL = svds(R.T @ np.diag(C @ np.ones(3)) @ L, k=3)
        UR = np.linalg.qr(R @ VR)[0]
        UL = np.linalg.qr(L @ VL)[0]
        
    elif option == 'pca':
        UR, _, _ = svds(R, k=3)
        UL, _, _ = svds(L, k=3)
        UR = np.linalg.qr(UR)[0]
        UL = np.linalg.qr(UL)[0]
        
    elif option == 'wpca':
        R_total = np.hstack([R * C[:, i:i+1] for i in range(3)])
        L_total = np.hstack([L * C[:, i:i+1] for i in range(3)])
        
        UR, _, _ = svds(R_total, k=3)
        UL, _, _ = svds(L_total, k=3)
        
        UR = np.linalg.qr(UR)[0]
        UL = np.linalg.qr(UL)[0]
        
    elif option == 'wpca1':
        UR = np.zeros((R.shape[0], 3))
        UL = np.zeros((L.shape[0], 3))
        
        for i in range(3):
            # Fix: Properly handle broadcasting and reshape
            R_weighted = R * np.tile(C[:, i:i+1], (1, R.shape[1]))
            L_weighted = L * np.tile(C[:, i:i+1], (1, L.shape[1]))
            
            U_temp, _, _ = svds(R_weighted, k=1)
            UR[:, i] = U_temp.flatten()
            
            U_temp, _, _ = svds(L_weighted, k=1)
            UL[:, i] = U_temp.flatten()
        
        UR = np.linalg.qr(UR)[0]
        UL = np.linalg.qr(UL)[0]
    
    return UR, UL