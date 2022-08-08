import cv2
import numpy as np
import math as m
import sys
from scipy.special import gamma as tgamma
import os
import csv
import time
# import svm functions (from libsvm library)   
if sys.version_info[0] < 3:
    import svm
    import svmutil
    from svmutil import *
    from svm import *
else:
    import svm
    import svmutil
    from svm import *
    from svmutil import *

class BRISQUE:
    def __init__(self):
            pass


    def AGGDfit(self, structdis):
        """AGGD 拟合模型,将输入作为 MSCN Image

            input: MSCN image
            output: 最佳拟合参数
        """
        # 用于计算正像素/负像素及其平方和的变量
        poscount = 0
        negcount = 0
        possqsum = 0
        negsqsum = 0
        abssum   = 0

        poscount = len(structdis[structdis > 0]) # 正像素数量
        negcount = len(structdis[structdis < 0]) # 负像素数量
        
        possqsum = np.sum(np.power(structdis[structdis > 0], 2))
        negsqsum = np.sum(np.power(structdis[structdis < 0], 2))
        
        # 绝对平方和
        abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

        # calculate left sigma variance and right sigma variance
        lsigma_best = np.sqrt((negsqsum/negcount))
        rsigma_best = np.sqrt((possqsum/poscount))

        gammahat = lsigma_best/rsigma_best
        
        # 像素总数
        totalcount = structdis.shape[1] * structdis.shape[0]

        rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
        rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))
        
        prevgamma = 0
        prevdiff  = 1e10
        sampling  = 0.001
        gam = 0.2

        #用于最佳拟合参数的向量化函数调用
        vectfunc = np.vectorize(self.func, otypes = [np.float], cache = False)
        
        gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

        return [lsigma_best, rsigma_best, gamma_best] 

    def func(self, gam, prevgamma, prevdiff, sampling, rhatnorm):
        """
        计算最佳拟合参数
        input: 一些参数
        output: 最佳拟合参数
        """
        while(gam < 10):
            r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
            diff = abs(r_gam - rhatnorm)
            if(diff > prevdiff): break
            prevdiff = diff
            prevgamma = gam
            gam += sampling
        gamma_best = prevgamma
        return gamma_best

    def compute_features(self, img):
        """
        计算特征向量

        input: 输入特征
        output: 特征向量的18个元素
        """
        scalenum = 2
        feat = []
        im_original = img.copy()

        for itr_scale in range(scalenum):
            im = im_original.copy()
            im = im / 255.0

            # 计算MSCN系数
            mu = cv2.GaussianBlur(im, (7, 7), 1.166)
            mu_sq = mu * mu

            sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
            sigma = (sigma - mu_sq) ** 0.5
            
            # structdis 表示MSCN图像
            structdis = im - mu
            structdis /= (sigma + 1.0/255)
            
            # 计算MSCN图像的最佳拟合参数
            best_fit_params = self.AGGDfit(structdis)

            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]
            
            # 为 MSCN 图像附加最佳拟合参数
            feat.append(gamma_best)
            feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

            # 计算相邻像素间的乘积关系
            shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

            for itr_shift in range(1, len(shifts) + 1):
                OrigArr = structdis
                reqshift = shifts[itr_shift-1] 

                # 创建变换矩阵
                M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
                ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
                
                Shifted_new_structdis = ShiftArr
                Shifted_new_structdis = Shifted_new_structdis * structdis
                # shifted_new_structdis 成对积
                best_fit_params = self.AGGDfit(Shifted_new_structdis)
                lsigma_best = best_fit_params[0]
                rsigma_best = best_fit_params[1]
                gamma_best  = best_fit_params[2]

                constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
                meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant


                feat.append(gamma_best) #形状
                feat.append(meanparam) #均值
                feat.append(m.pow(lsigma_best, 2)) # 左方差平方
                feat.append(m.pow(rsigma_best, 2)) # 右方差平方
            
            # 在下一次迭代中调整图像大小
            im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        return feat

    def test_measure_BRISQUE(self, imgPath):
        """
        计算BRISQUE 的质量分数
        input: 图像路径
        output: 质量分数
        """
        dis = cv2.imread(imgPath, 1)
        if(dis is None):
            print("Wrong image path given")
            print("Exiting...")
            sys.exit(0)

        dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

        # 计算特征向量
        features = self.compute_features(dis)

        #将 brisqueFeatures 向量重新缩放到 [-1,1]
        x = [0]
        
        # 通过 C++ 模块预加载列表以将 brisquefeatures 向量重新缩放为 [-1, 1]
        min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
        
        max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

        # 将重新缩放的向量附加到 x
        for i in range(0, 36):
            min = min_[i]
            max = max_[i] 
            x.append(-1 + (2.0/(max - min) * (features[i] - min)))
        
        # 加载模型
        model = svmutil.svm_load_model("allmodel")

        #从 python 列表创建 svm 节点数组
        x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
        x[36].index = -1 #将最后一个索引设置为 -1 以指示结束
        
        # 获取重要参数
        svm_type = model.get_svm_type()
        
        if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
            # 回归问题设置svm_type 是 EPSILON_SVR
            nr_classifier = 1
        dec_values = (c_double * nr_classifier)()
        
        # 使用模型和 svm_node_array 计算图像的质量得分
        qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)
        return qualityscore