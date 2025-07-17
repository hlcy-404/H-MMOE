import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaseExpert(nn.Module):
    """专家基类"""
    def __init__(self, emb_size, shared_encoder=None):
        super().__init__()
        self.emb_size = emb_size
        self.shared_encoder = shared_encoder
        
    def _init_weights(self):
        """
        根据专家使用的激活函数类型初始化线性层权重
        """
        # 检测序列中使用的激活函数类型
        activation_type = None
        activation_instance = None
        
        for module in self.seq:
            if isinstance(module, (nn.SiLU, nn.GELU, nn.ELU, nn.LeakyReLU)) or \
               (hasattr(module, '__class__') and module.__class__.__name__ == 'Mish'):
                activation_type = type(module)
                activation_instance = module
                break
                
        # 定义初始化函数
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 根据激活函数类型选择合适的初始化
                if activation_type is None:
                    # 默认Xavier初始化
                    nn.init.xavier_uniform_(m.weight)
                elif activation_type == nn.SiLU or activation_type == nn.GELU:
                    # SiLU/GELU激活函数初始化 - 类似ReLU特性
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif activation_type == nn.ELU:
                    # ELU激活函数初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif activation_type == nn.LeakyReLU:
                    # LeakyReLU激活函数初始化
                    leaky_slope = activation_instance.negative_slope
                    nn.init.kaiming_normal_(
                        m.weight, 
                        a=leaky_slope, 
                        mode='fan_in', 
                        nonlinearity='leaky_relu'
                    )
                elif activation_instance.__class__.__name__ == 'Mish':
                    # Mish激活函数初始化 - 类似ReLU特性
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    
                # 偏置项统一初始化为0
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 应用初始化到序列中的所有层
        # self.seq.apply(init_weights)
        print(f"Applied weight initialization for {self.__class__.__name__} based on {activation_instance.__class__.__name__ if activation_instance else 'default'}")
        
    def forward(self, x, **kwargs):
        """
        默认优先处理图像和掩码，仅在处理失败时回退到特征处理
        
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 默认实现，子类应当覆盖此方法
        return x

class StandardExpert(BaseExpert):
    """标准专家，现在也支持图像和掩码输入，与滤波专家保持一致的参数量"""
    def __init__(self, emb_size, config=None, shared_encoder=None):
        super().__init__(emb_size, shared_encoder)
        # 如果没有提供共享编码器，则创建自己的编码器
        if self.shared_encoder is None:
            # 添加ResNet18特征提取器
            from torchvision.models import resnet18
            from torchvision import models
            self.private_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.private_encoder.fc.in_features
            self.output = int(emb_size / 4)
            self.private_encoder.fc = nn.Linear(num_ftrs, self.output)
            self.private_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.private_encoder = None
            
        # 定义编码器访问方法
        self.get_encoder = lambda: self.shared_encoder if self.shared_encoder is not None else self.private_encoder
        
        # FNN处理部分保持不变
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),
            nn.Linear(emb_size, emb_size),
            nn.SiLU(),

        )
        
        # 移除权重初始化
        # self._init_weights()
    
    def _process_image(self, image):
        """将输入图像处理为单通道"""
        if len(image.shape) == 4:  # [B,C,H,W]
            batch_results = []
            for i in range(image.shape[0]):
                img = image[i].permute(1,2,0).cpu().numpy()  # [H,W,C]
                
                # 转为灰度图
                if img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.squeeze()
                
                # 归一化
                gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX)
                
                # 转回tensor格式 [1,H,W]
                gray_tensor = torch.from_numpy(gray).unsqueeze(0).to(image.device)
                batch_results.append(gray_tensor)
            
            return torch.stack(batch_results)
        else:
            raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    def forward(self, x, input_v=None, masks=None, **kwargs):
        """
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 获取当前应该使用的编码器
        encoder = self.get_encoder()
        
        # 检查input_v是否是图像格式(4D张量[B,C,H,W])
        is_image_input = input_v is not None and len(input_v.shape) == 4
        is_mask_valid = masks is not None and all(isinstance(m, torch.Tensor) and len(m.shape) == 4 for m in masks)
        
        # 默认优先处理图像
        if is_image_input and is_mask_valid and encoder is not None:
            try:
                # 处理输入图像和掩码为单通道
                proc_input_v = self._process_image(input_v)
                proc_source = self._process_image(masks[0])
                proc_target = self._process_image(masks[1])
                proc_background = self._process_image(masks[2])
                
                # 特征提取
                s = encoder(proc_source)
                t = encoder(proc_target)
                b = encoder(proc_background)
                v = encoder(proc_input_v)
                
                # 特征融合
                visionFeatures = torch.cat((s, t, b, v), dim=1)
                
                # FNN处理
                return self.seq(visionFeatures)
            except Exception as e:
                # 如果图像处理失败，回退到特征处理模式
                print(f"Warning: Image processing failed in StandardExpert: {e}. Using feature mode.")
        
        # 回退模式：直接处理特征向量
        return self.seq(x)

class SobelExpert(BaseExpert):
    """Sobel滤波器专家 - 空间边缘专家"""
    def __init__(self, emb_size, config, shared_encoder=None):
        super().__init__(emb_size, shared_encoder)
        self.config = config
        
        # 如果没有提供共享编码器，则创建自己的编码器
        if self.shared_encoder is None:
            # 使用ResNet18提取特征
            from torchvision.models import resnet18
            from torchvision import models
            self.private_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.private_encoder.fc.in_features
            self.output = int(emb_size / 4)
            self.private_encoder.fc = nn.Linear(num_ftrs, self.output)
            self.private_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.private_encoder = None
        
        # 定义编码器访问方法
        self.get_encoder = lambda: self.shared_encoder if self.shared_encoder is not None else self.private_encoder
        
        # 使用Mish激活函数 - 更适合边缘特征
        class Mish(nn.Module):
            def forward(self, x):
                return x * torch.tanh(F.softplus(x))
        
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            Mish(),
            nn.Linear(emb_size, emb_size),
            Mish(),

        )
        
        # 移除权重初始化
        # self._init_weights()
        
    def _apply_sobel(self, image):
        """应用Sobel滤波器"""
        # 确保图像适合处理
        if len(image.shape) == 4:  # [B,C,H,W]
            batch_results = []
            for i in range(image.shape[0]):
                img = image[i].permute(1,2,0).cpu().numpy()  # [H,W,C]
                
                # 转为灰度图
                if img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.squeeze()
                
                # 应用Sobel
                sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                
                # 合并结果
                magnitude = cv2.magnitude(sobel_x, sobel_y)
                
                # 归一化
                magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
                
                # 转回tensor格式 [1,H,W]
                magnitude_tensor = torch.from_numpy(magnitude).unsqueeze(0).to(image.device)
                batch_results.append(magnitude_tensor)
            
            return torch.stack(batch_results)
        else:
            raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    def forward(self, x, input_v=None, masks=None, **kwargs):
        """
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 获取当前应该使用的编码器
        encoder = self.get_encoder()
        
        # 检查input_v是否是图像格式(4D张量[B,C,H,W])
        is_image_input = input_v is not None and len(input_v.shape) == 4
        is_mask_valid = masks is not None and all(isinstance(m, torch.Tensor) and len(m.shape) == 4 for m in masks)
        
        # 默认优先处理图像
        if is_image_input and is_mask_valid and encoder is not None:
            try:
                # 应用Sobel滤波
                sobel_input_v = self._apply_sobel(input_v)
                sobel_source = self._apply_sobel(masks[0])
                sobel_target = self._apply_sobel(masks[1])
                sobel_background = self._apply_sobel(masks[2])
                
                # 特征提取
                s = encoder(sobel_source)
                t = encoder(sobel_target)
                b = encoder(sobel_background)
                v_filt = encoder(sobel_input_v)
                
                # 特征融合
                visionFeatures = torch.cat((s, t, b, v_filt), dim=1)
                
                # FNN处理
                return self.seq(visionFeatures)
            except Exception as e:
                # 如果出错，回退到特征处理模式
                print(f"Warning: Image processing failed in SobelExpert: {e}. Using feature mode.")
        
        # 回退到特征处理模式
        return self.seq(x)

class SchmidExpert(BaseExpert):
    """Schmid滤波器专家 - 空间纹理专家"""
    def __init__(self, emb_size, config, shared_encoder=None):
        super().__init__(emb_size, shared_encoder)
        self.config = config
        
        # 如果没有提供共享编码器，则创建自己的编码器
        if self.shared_encoder is None:
            from torchvision.models import resnet18
            from torchvision import models
            self.private_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.private_encoder.fc.in_features
            self.output = int(emb_size / 4)
            self.private_encoder.fc = nn.Linear(num_ftrs, self.output)
            self.private_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.private_encoder = None
            
        # 定义编码器访问方法
        self.get_encoder = lambda: self.shared_encoder if self.shared_encoder is not None else self.private_encoder
        
        # 使用ELU激活函数 - 适合纹理特征
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ELU(),
            nn.Linear(emb_size, emb_size),
            nn.ELU(),

        )
        
        # 预计算多个Schmid核，捕获不同尺度纹理
        self.kernels = self._create_schmid_kernels()
        
        # 初始化权重
        # self._init_weights()
        
    def _create_schmid_kernels(self):
        """创建多个Schmid滤波核"""
        # 不同的tao和sigma组合捕获不同尺度纹理
        params = [
            (2, 1),  # 细纹理
            (4, 2),  # 中等纹理
            (8, 4),  # 粗纹理
        ]
        
        kernels = []
        for tao, sigma in params:
            kernel = self._schmid_kernel(tao, sigma)
            # 转为PyTorch格式的卷积核
            kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
            kernels.append(kernel_tensor)
            
        return kernels
        
    def _schmid_kernel(self, tao, sigma):
        """生成Schmid滤波核"""
        sigma2 = float(sigma * sigma)
        half_filter_size = 10
        filter_size = half_filter_size * 2 + 1
        
        schmid = np.zeros((filter_size, filter_size), np.float32)
        
        for i in range(filter_size):
            for j in range(filter_size):
                x = i - half_filter_size
                y = j - half_filter_size
                r2 = float(x * x + y * y)
                
                if r2 <= half_filter_size * half_filter_size:
                    schmid[i, j] = np.exp(-r2 / (2 * sigma2)) * np.cos(np.pi * r2 / tao)
        
        # 归一化
        sum_val = np.sum(schmid)
        if sum_val != 0:
            schmid = schmid / sum_val
        
        return schmid
        
    def _apply_schmid(self, image):
        """应用Schmid滤波器"""
        if len(image.shape) == 4:  # [B,C,H,W]
            batch_results = []
            for i in range(image.shape[0]):
                img = image[i].permute(1,2,0).cpu().numpy()  # [H,W,C]
                
                # 转为灰度图
                if img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.squeeze()
                
                # 应用所有Schmid核并取最大响应
                response = None
                for kernel_tensor in self.kernels:
                    kernel = kernel_tensor.squeeze().cpu().numpy()
                    curr_response = cv2.filter2D(gray, -1, kernel)
                    if response is None:
                        response = curr_response
                    else:
                        response = np.maximum(response, curr_response)
                
                # 归一化
                response = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)
                
                # 转回tensor格式 [1,H,W]
                response_tensor = torch.from_numpy(response).unsqueeze(0).to(image.device)
                batch_results.append(response_tensor)
            
            return torch.stack(batch_results)
        else:
            raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    def forward(self, x, input_v=None, masks=None, **kwargs):
        """
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 获取当前应该使用的编码器
        encoder = self.get_encoder()
        
        # 检查input_v是否是图像格式(4D张量[B,C,H,W])
        is_image_input = input_v is not None and len(input_v.shape) == 4
        is_mask_valid = masks is not None and all(isinstance(m, torch.Tensor) and len(m.shape) == 4 for m in masks)
        
        # 默认优先处理图像
        if is_image_input and is_mask_valid and encoder is not None:
            try:
                # 应用Schmid滤波
                schmid_input_v = self._apply_schmid(input_v)
                schmid_source = self._apply_schmid(masks[0])
                schmid_target = self._apply_schmid(masks[1])
                schmid_background = self._apply_schmid(masks[2])
                
                # 特征提取
                s = encoder(schmid_source)
                t = encoder(schmid_target)
                b = encoder(schmid_background)
                v_filt = encoder(schmid_input_v)
                
                # 特征融合
                visionFeatures = torch.cat((s, t, b, v_filt), dim=1)
                
                # FNN处理
                return self.seq(visionFeatures)
            except Exception as e:
                # 如果出错，回退到特征处理模式
                print(f"Warning: Image processing failed in SchmidExpert: {e}. Using feature mode.")
        
        # 回退到特征处理模式
        return self.seq(x)

class GaborExpert(BaseExpert):
    """Gabor滤波器专家 - 方向和频率特征专家"""
    def __init__(self, emb_size, config, shared_encoder=None):
        super().__init__(emb_size, shared_encoder)
        self.config = config
        
        # 如果没有提供共享编码器，则创建自己的编码器
        if self.shared_encoder is None:
            from torchvision.models import resnet18
            from torchvision import models
            self.private_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.private_encoder.fc.in_features
            self.output = int(emb_size / 4)
            self.private_encoder.fc = nn.Linear(num_ftrs, self.output)
            self.private_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.private_encoder = None
        
        # 定义编码器访问方法
        self.get_encoder = lambda: self.shared_encoder if self.shared_encoder is not None else self.private_encoder
        
        # GELU激活函数 - 适合捕获Gabor特征
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),

        )
        
        # 预计算多个Gabor核，捕获不同方向和频率的特征
        self.kernels = self._create_gabor_kernels()
        
        # 初始化权重
        # self._init_weights()
        
    def _create_gabor_kernels(self):
        """创建多个Gabor滤波核"""
        # 不同的方向和频率组合
        orientations = [0, 45, 90, 135]  # 度数
        frequencies = [0.1, 0.25, 0.4]  # 相对频率
        
        kernels = []
        for theta in orientations:
            for freq in frequencies:
                theta_rad = theta * np.pi / 180  # 转换为弧度
                # 创建Gabor滤波核
                kernel_size = 15
                sigma = 4.0
                lambd = 10.0 / freq
                gamma = 0.5
                psi = 0
                
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma, theta_rad, lambd, gamma, psi, ktype=cv2.CV_32F
                )
                
                # 归一化
                kernel = kernel / np.sum(np.abs(kernel))
                
                # 转为PyTorch格式的卷积核
                kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
                kernels.append(kernel_tensor)
                
        return kernels
        
    def _apply_gabor(self, image):
        """应用Gabor滤波器"""
        if len(image.shape) == 4:  # [B,C,H,W]
            batch_results = []
            for i in range(image.shape[0]):
                img = image[i].permute(1,2,0).cpu().numpy()  # [H,W,C]
                
                # 转为灰度图
                if img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.squeeze()
                
                # 应用所有Gabor核并取最大响应
                response = None
                for kernel_tensor in self.kernels:
                    kernel = kernel_tensor.squeeze().cpu().numpy()
                    curr_response = cv2.filter2D(gray, -1, kernel)
                    if response is None:
                        response = curr_response
                    else:
                        response = np.maximum(response, curr_response)
                
                # 归一化
                response = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)
                
                # 转回tensor格式 [1,H,W]
                response_tensor = torch.from_numpy(response).unsqueeze(0).to(image.device)
                batch_results.append(response_tensor)
            
            return torch.stack(batch_results)
        else:
            raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    def forward(self, x, input_v=None, masks=None, **kwargs):
        """
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 获取当前应该使用的编码器
        encoder = self.get_encoder()
        
        # 检查input_v是否是图像格式(4D张量[B,C,H,W])
        is_image_input = input_v is not None and len(input_v.shape) == 4
        is_mask_valid = masks is not None and all(isinstance(m, torch.Tensor) and len(m.shape) == 4 for m in masks)
        
        # 默认优先处理图像
        if is_image_input and is_mask_valid and encoder is not None:
            try:
                # 应用Gabor滤波
                gabor_input_v = self._apply_gabor(input_v)
                gabor_source = self._apply_gabor(masks[0])
                gabor_target = self._apply_gabor(masks[1])
                gabor_background = self._apply_gabor(masks[2])
                
                # 特征提取
                s = encoder(gabor_source)
                t = encoder(gabor_target)
                b = encoder(gabor_background)
                v_filt = encoder(gabor_input_v)
                
                # 特征融合
                visionFeatures = torch.cat((s, t, b, v_filt), dim=1)
                
                # FNN处理
                return self.seq(visionFeatures)
            except Exception as e:
                # 如果出错，回退到特征处理模式
                print(f"Warning: Image processing failed in GaborExpert: {e}. Using feature mode.")
        
        # 回退到特征处理模式
        return self.seq(x)

class WaveletExpert(BaseExpert):
    """小波变换专家 - 多尺度分析专家"""
    def __init__(self, emb_size, config, shared_encoder=None):
        super().__init__(emb_size, shared_encoder)
        self.config = config
        
        # 如果没有提供共享编码器，则创建自己的编码器
        if self.shared_encoder is None:
            from torchvision.models import resnet18
            from torchvision import models
            self.private_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.private_encoder.fc.in_features
            self.output = int(emb_size / 4)
            self.private_encoder.fc = nn.Linear(num_ftrs, self.output)
            self.private_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.private_encoder = None
        
        # 定义编码器访问方法
        self.get_encoder = lambda: self.shared_encoder if self.shared_encoder is not None else self.private_encoder
        
        # 使用LeakyReLU - 适合小波特征
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(0.2),

        )
        
        # 初始化权重
        # self._init_weights()
        
    def _apply_wavelet(self, image):
        """应用小波变换"""
        if len(image.shape) == 4:  # [B,C,H,W]
            batch_results = []
            for i in range(image.shape[0]):
                img = image[i].permute(1,2,0).cpu().numpy()  # [H,W,C]
                
                # 转为灰度图
                if img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img.squeeze()
                
                # 简化的小波变换 - 使用拉普拉斯金字塔作为近似
                # 原始尺寸
                orig_h, orig_w = gray.shape
                
                # 创建不同尺度的表示
                level1 = cv2.pyrDown(gray)
                level2 = cv2.pyrDown(level1)
                
                # 重建金字塔
                level1_up = cv2.pyrUp(level1, dstsize=(orig_w, orig_h))
                level2_up = cv2.pyrUp(cv2.pyrUp(level2), dstsize=(orig_w, orig_h))
                
                # 计算细节系数
                detail1 = cv2.subtract(gray, level1_up)
                detail2 = cv2.subtract(level1_up, level2_up)
                
                def soft_threshold(x, threshold):
                    """软阈值处理 - 去噪"""
                    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
                
                # 软阈值处理 - 去噪
                detail1 = soft_threshold(detail1, 0.1)
                detail2 = soft_threshold(detail2, 0.05)
                
                # 融合不同尺度的特征
                wavelet_response = detail1 * 0.7 + detail2 * 0.3
                
                # 归一化
                wavelet_response = cv2.normalize(wavelet_response, None, 0, 1, cv2.NORM_MINMAX)
                
                # 转回tensor格式 [1,H,W]
                response_tensor = torch.from_numpy(wavelet_response).unsqueeze(0).to(image.device)
                batch_results.append(response_tensor)
            
            return torch.stack(batch_results)
        else:
            raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    def forward(self, x, input_v=None, masks=None, **kwargs):
        """
        Args:
            x: 视觉特征 Fv
            input_v: 原始图像
            masks: 掩码列表 [source, target, background]
        """
        # 获取当前应该使用的编码器
        encoder = self.get_encoder()
        
        # 检查input_v是否是图像格式(4D张量[B,C,H,W])
        is_image_input = input_v is not None and len(input_v.shape) == 4
        is_mask_valid = masks is not None and all(isinstance(m, torch.Tensor) and len(m.shape) == 4 for m in masks)
        
        # 默认优先处理图像
        if is_image_input and is_mask_valid and encoder is not None:
            try:
                # 应用小波变换
                wavelet_input_v = self._apply_wavelet(input_v)
                wavelet_source = self._apply_wavelet(masks[0])
                wavelet_target = self._apply_wavelet(masks[1])
                wavelet_background = self._apply_wavelet(masks[2])
                
                # 特征提取
                s = encoder(wavelet_source)
                t = encoder(wavelet_target)
                b = encoder(wavelet_background)
                v_filt = encoder(wavelet_input_v)
                
                # 特征融合
                visionFeatures = torch.cat((s, t, b, v_filt), dim=1)
                
                # FNN处理
                return self.seq(visionFeatures)
            except Exception as e:
                # 如果出错，回退到特征处理模式
                print(f"Warning: Image processing failed in WaveletExpert: {e}. Using feature mode.")
        
        # 回退到特征处理模式
        return self.seq(x) 