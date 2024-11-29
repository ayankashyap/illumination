import scipy
import cv2
import numpy as np

from PIL import Image
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from cv2 import ximgproc

from utils import compute_base, gamma_correct, norm_img, to_8bit
from two_light import est_two_light_coeff

# def _load_data(constants_pth, im_pth):
#     constants_dict= scipy.io.loadmat(constants_pth)
#     data_dict = scipy.io.loadmat(im_pth)
#     return constants_dict, data_dict

    
class LightSourceSolver:
    
    def __init__(self, constants_pth: str):
        constants = scipy.io.loadmat(constants_pth)
        self.L = constants['L']
        self.R = constants['R']
        self.C = constants['C']
        # compute reflectance and illumination bases
        self.Rb, self.Lb = compute_base(L=self.L, C=self.C, R=self.R, option='wpca1')
            
        E = np.zeros((3, 3, 3))
        for ind in range(3):
            # Compute E for each color channel
            E[:,:,ind] = self.Rb.T @ (self.C[:,ind:ind+1] * self.Lb)

        self.E = E
        
        # helper params
        f = self.Lb.T @ (0.025 * np.ones_like(self.L[:, 0]))
        self.f = f / np.linalg.norm(f)
        self.lambda_val = 1e-5
        self.cutoff = 0.1

    def solve(self, im_pth, illum_num):
        assert illum_num in [2,3]
        results = {}
        im_dict = scipy.io.loadmat(im_pth)

        im_nf = im_dict['im_nf']
        im_f = im_dict['im_f']
        mask = im_dict['mask']
        im_size = im_nf.shape
        
        # pure flash image
        diff_img = np.maximum(im_f - im_nf, 0) * np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        
        # compute alphas
        Amat = np.zeros((3, 3))
        for i in range(3):
            Amat[:, i] = self.E[:,:,i] @ self.f

        diff_img_vec = diff_img.reshape(-1, 3).T
        alpha = np.linalg.lstsq(Amat.T, diff_img_vec, rcond=None)[0]
        norm_alpha = np.sqrt(np.sum(alpha**2, axis=0))
        results['alpha'] = alpha.T.reshape(im_size)
        results['norm_alpha'] = norm_alpha.reshape(im_size[:2])

        # compute beta
        img_nf_vec = im_nf.reshape(-1, 3).T
        beta = np.zeros_like(alpha)
        beta_norm = np.zeros(alpha.shape[1])
        gamma = np.zeros_like(alpha)
        print("alpha", alpha.shape) 
        # Solve for beta and gamma
        for kk in range(diff_img_vec.shape[1]):
            if kk % 10000 == 0:
                print(f'Processing pixel {kk}/{diff_img_vec.shape[1]}')
                
            if mask.flat[kk]:
                Bmat = np.zeros((3, 3))
                for i in range(3):
                    Bmat[i] = alpha[:, kk].T @ self.E[:,:,i]
                Bmat = Bmat / (1e-10 + norm_alpha[kk])
                
                nf_intensity = img_nf_vec[:, kk]
                
                beta[:, kk] = np.linalg.solve(
                    Bmat.T @ Bmat + self.lambda_val * np.eye(3),
                    Bmat.T @ nf_intensity
                )
                
                if np.any(np.isnan(beta[:, kk])):
                    beta[:, kk] = 0

        im_nf_t = im_nf * np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        beta_img = beta.T.reshape(im_size)
        smoothness = 0.0000001
        for i in range(3):
            beta_img[:,:,i] = ximgproc.guidedFilter(
                im_nf_t[:,:,i].astype(np.float32),
                beta_img[:,:,i].astype(np.float32),
                5, smoothness
            )        

        beta_img = beta_img.reshape(-1, 3).T
        beta_norm = np.sqrt(np.sum(beta_img**2, axis=0))
        gamma = beta_img / (np.repeat(beta_norm[np.newaxis, :], 3, axis=0) + 1e-10)

        results['beta'] = beta.T.reshape(im_size)
        results['beta_norm'] = beta_norm.reshape(im_size[:2])
        results['gamma'] = gamma.T.reshape(im_size)

        if illum_num == 2:
            # Two light case
            illum1, illum2 = est_two_light_coeff(gamma, mask, self.cutoff)
            # calculate relative shading
            mR1 = (alpha.T @ self.E[:,:,0] @ illum1) * (beta_norm / (1e-10 + norm_alpha))
            mR2 = (alpha.T @ self.E[:,:,0] @ illum2) * (beta_norm / (1e-10 + norm_alpha))
            mG1 = (alpha.T @ self.E[:,:,1] @ illum1) * (beta_norm / (1e-10 + norm_alpha))
            mG2 = (alpha.T @ self.E[:,:,1] @ illum2) * (beta_norm / (1e-10 + norm_alpha))
            mB1 = (alpha.T @ self.E[:,:,2] @ illum1) * (beta_norm / (1e-10 + norm_alpha))
            mB2 = (alpha.T @ self.E[:,:,2] @ illum2) * (beta_norm / (1e-10 + norm_alpha))

            coeff = np.zeros((2, diff_img_vec.shape[1]))
            for kk in range(diff_img_vec.shape[1]):
                Amat_ = np.array([[mR1[kk], mR2[kk]],
                                [mG1[kk], mG2[kk]],
                                [mB1[kk], mB2[kk]]])
                bvec_ = img_nf_vec[:, kk]
                coeff[:, kk] = np.linalg.lstsq(Amat_, bvec_, rcond=None)[0]

            # Calculate separated images
            corr1 = coeff[0] * beta_norm / (1e-10 + norm_alpha)
            corr2 = coeff[1] * beta_norm / (1e-10 + norm_alpha)
            
            # Generate final images
            im1_new = np.zeros(im_size)
            im2_new = np.zeros(im_size)
            im1_wb = np.zeros(im_size)
            im2_wb = np.zeros(im_size)
            
            for k in range(3):
                im1_new[..., k] = (alpha.T @ self.E[:,:,k] @ illum1 * corr1).reshape(im_size[:2])
                im2_new[..., k] = (alpha.T @ self.E[:,:,k] @ illum2 * corr2).reshape(im_size[:2])
                im1_wb[..., k] = (alpha.T @ self.E[:,:,k] @ self.f * corr1).reshape(im_size[:2])
                im2_wb[..., k] = (alpha.T @ self.E[:,:,k] @ self.f * corr2).reshape(im_size[:2])
            
            results.update({
                'im1': im1_new,
                'im2': im2_new,
                'im1_wb': im1_wb,
                'im2_wb': im2_wb,
                'coeff': coeff
            })
        
        else:
            raise NotImplementedError
        
        return results, im_dict
    

if __name__ == "__main__":
    
    ls_solver = LightSourceSolver(constants_pth="illuminant_separation/data/reflectance_illum_camera.mat")
    results, im_dict = ls_solver.solve(im_pth="illuminant_separation/data/images/two_lights/conference_room.mat", illum_num=2)

    np.save("im_nf.npy", im_dict['im_nf'])
    np.save("im_1.npy", results['im1'])
    np.save("im_2.npy", results['im2']) 
    im_nf = to_8bit(gamma_correct(im_dict['im_nf']))
    im1 = to_8bit(gamma_correct(np.abs(results['im1'])))
    im2 = to_8bit(gamma_correct(np.abs(results['im2'])))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(im_nf)
    plt.title('No-flash image')
    
    plt.subplot(132)
    plt.imshow(im1)
    plt.title('Separated light 1')
    
    plt.subplot(133)
    plt.imshow(im2)
    plt.title('Separated light 2')
    
    plt.show()
        


        

        

        
        

        