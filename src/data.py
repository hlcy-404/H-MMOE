import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from .Loader import Loader
from .seqEncoder import SeqEncoder

def get_transforms(config):
    """获取数据转换函数"""
    image_size = config["image_resize"]
    
    # 图像转换
    image_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 掩码转换
    mask_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    return {
        "image": image_transform,
        "mask": mask_transform
    }

def get_test_loader(config):
    """
    获取测试数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        test_loader: 测试数据加载器
    """
    try:
        # 获取数据转换
        transforms = get_transforms(config)
        
        # 创建序列编码器
        seq_encoder = SeqEncoder(config)
        
        # 创建测试数据集
        test_dataset = Loader(
            config=config,
            DataConfig=config["TestConfig"],
            seqEncoder=seq_encoder,
            img_size=config["image_resize"],
            textHead=config["textHead"],
            imageHead=config["imageHead"],
            train=False,
            transform=transforms
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False)
        )
        
        return test_loader
    
    except Exception as e:
        print(f"创建测试数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_val_loader(config):
    """
    获取验证数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        val_loader: 验证数据加载器
    """
    try:
        # 获取数据转换
        transforms = get_transforms(config)
        
        # 创建序列编码器
        seq_encoder = SeqEncoder(config)
        
        # 创建验证数据集
        val_dataset = Loader(
            config=config,
            DataConfig=config["ValConfig"],
            seqEncoder=seq_encoder,
            img_size=config["image_resize"],
            textHead=config["textHead"],
            imageHead=config["imageHead"],
            train=False,
            transform=transforms
        )
        
        # 创建数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False)
        )
        
        return val_loader
    
    except Exception as e:
        print(f"创建验证数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_train_loader(config):
    """
    获取训练数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        train_loader: 训练数据加载器
    """
    try:
        # 获取数据转换
        transforms = get_transforms(config)
        
        # 创建序列编码器
        seq_encoder = SeqEncoder(config)
        
        # 创建训练数据集
        train_dataset = Loader(
            config=config,
            DataConfig=config["TrainConfig"],
            seqEncoder=seq_encoder,
            img_size=config["image_resize"],
            textHead=config["textHead"],
            imageHead=config["imageHead"],
            train=True,
            transform=transforms
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False)
        )
        
        return train_loader
    
    except Exception as e:
        print(f"创建训练数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None 