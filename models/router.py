import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from .modules import CrossAttention
import torch.nn.functional as F
from .experts import StandardExpert, SobelExpert, SchmidExpert, GaborExpert, WaveletExpert


# Mixture of Experts
class MoE(nn.Module):
    def __init__(self, config, expert_types, top, emb_size, w_importance=0.01):
        super().__init__()
        self.attConfig = config["attnConfig"]
        self.expert_types = expert_types
        self.num_experts = len(expert_types)
        self.top = top
        self.experts = nn.ModuleList()
        
        # 移除共享编码器的创建
        self.shared_encoder = None
        
        # 创建专家，传入None作为shared_encoder参数，强制每个专家创建自己的编码器
        for expert_type in expert_types:
            if expert_type == 'standard':
                self.experts.append(StandardExpert(emb_size, config, shared_encoder=None))
            elif expert_type == 'sobel':
                self.experts.append(SobelExpert(emb_size, config, shared_encoder=None))
            elif expert_type == 'schmid':
                self.experts.append(SchmidExpert(emb_size, config, shared_encoder=None))
            elif expert_type == 'gabor':
                self.experts.append(GaborExpert(emb_size, config, shared_encoder=None))
            elif expert_type == 'wavelet':
                self.experts.append(WaveletExpert(emb_size, config, shared_encoder=None))
        
        self.crossAtt = CrossAttention(
            self.attConfig["embed_size"],
            self.attConfig["heads"],
            self.attConfig["attn_dropout"],
        )
        self.gate = nn.Linear(emb_size, self.num_experts)
        self.noise = nn.Linear(emb_size, self.num_experts)  # add noise to gate
        self.w_importance = w_importance  # expert balance(for loss)

    def forward(self, x, q, input_v=None, masks=None):  # x: (batch,seq_len,emb)
        x_shape = x.shape

        x = x.reshape(-1, x_shape[-1])  # (batch*seq_len,emb)

        # gates
        att = self.crossAtt(x.unsqueeze(1), q.unsqueeze(1)).squeeze(1)
        gate_logits = self.gate(att)  # (batch*seq_len,experts)
        gate_prob = F.softmax(gate_logits, dim=-1)  # (batch*seq_len,experts)

        # 2024-05-05 Noisy Top-K Gating
        if self.training:
            noise = torch.randn_like(gate_prob) * F.softplus(
                self.noise(x))  # https://arxiv.org/pdf/1701.06538 , StandardNormal()*Softplus((x*W_noise))
            gate_prob = gate_prob + noise

        # top expert
        top_weights, top_index = torch.topk(gate_prob, k=self.top,
                                            dim=-1)  # top_weights: (batch*seq_len,top), top_index: (batch*seq_len,top)
        top_weights = F.softmax(top_weights, dim=-1)

        top_weights = top_weights.view(-1)  # (batch*seq_len*top)
        top_index = top_index.view(-1)  # (batch*seq_len*top)

        x = x.unsqueeze(1).expand(x.size(0), self.top, x.size(-1)).reshape(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = torch.zeros_like(x)  # (batch*seq_len*top,emb)

        # run by per expert
        for expert_i, expert_model in enumerate(self.experts):
            expert_indices = (top_index == expert_i).nonzero().flatten()
            if len(expert_indices) == 0:
                continue
                
            x_expert = x[expert_indices]  # (...,emb)
            
            # 无论专家类型都传入原始图像和掩码
            y_expert = expert_model(
                x_expert, 
                input_v=input_v, 
                masks=masks
            )

            y = y.index_add(dim=0, index=expert_indices,
                            source=y_expert)  # y[top_index==expert_i]=y_expert

        # weighted sum experts
        top_weights = top_weights.view(-1, 1).expand(-1, x.size(-1))  # (batch*seq_len*top,emb)
        y = y * top_weights
        y = y.view(-1, self.top, x.size(-1))  # (batch*seq_len,top,emb)
        y = y.sum(dim=1)  # (batch*seq_len,emb)

        # experts balance loss
        # https://arxiv.org/pdf/1701.06538 BALANCING EXPERT UTILIZATION
        if self.training:
            importance = gate_prob.sum(dim=0)  # sum( (batch*seq_len,experts) , dim=0)
            # Coefficient of Variation(CV), CV = standard deviation / mean
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
        else:
            importance = gate_prob.sum(dim=0)
            importance_loss = self.w_importance * (torch.std(importance) / torch.mean(importance)) ** 2
            # importance_loss = None
        return y.view(x_shape), gate_prob, importance_loss


class RouterGate(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(RouterGate, self).__init__()
        self.config = config
        self.embed_size = self.config["FUSION_IN"]
        self.attConfig = self.config["attnConfig"]
        self.output = int(self.attConfig["embed_size"] / 4)
        
        # 从配置获取专家类型，如果未指定则默认全部为标准专家
        self.expert_types = config.get("EXPERT_TYPES", ['standard'] * config["EXPERTS"])
        experts = len(self.expert_types)
        top = self.config["TOP"]
        
        # 创建MoE，它内部不再包含共享编码器
        self.moe = MoE(config, self.expert_types, top, self.attConfig["embed_size"])
        
        # 为RouterGate创建独立的编码器
        self.shared_encoder = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.shared_encoder.fc.in_features
        self.shared_encoder.fc = nn.Linear(num_ftrs, self.output)
        self.shared_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.linerImg = nn.Linear(self.attConfig["embed_size"], self.output)
        self.out = nn.Linear(int(self.embed_size * 2), self.embed_size)
        
        # 添加打印信息，说明使用了独立编码器
        print("\n======= 编码器配置 =======")
        print("RouterGate: 使用独立编码器")
        print(f"各专家: 使用独立编码器")
        print("=========================\n")

    def forward(self, source, target, background, image, text):
        # 使用共享编码器提取掩码特征
        s = self.shared_encoder(source)
        t = self.shared_encoder(target)
        b = self.shared_encoder(background)
        
        # 处理图像特征
        img = self.linerImg(image)
        img = nn.SiLU()(img)
        
        # 合并特征
        visionFeatures = torch.cat((s, t, b, img), dim=1)

        # 包含掩码列表
        masks = [source, target, background]
        
        # 调用MoE，传入原始图像和掩码
        moeFeatures, gate_prob, importance_loss = self.moe(
            visionFeatures, 
            text,
            input_v=image,  # 原始图像
            masks=masks     # 掩码图像列表
        )

        return moeFeatures, gate_prob, importance_loss
