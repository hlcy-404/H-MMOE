import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator  # 正确导入MultipleLocator

class MMoEVisualizer:
    """CM-MMoE模型特征可视化工具"""
    
    def __init__(self, save_dir="./tsne_visualizations"):
        """
        初始化可视化工具
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 扩展颜色列表以支持更多类别
        base_colors = ['#362635', '#3AB2B6', '#45A7D8', '#2F9194', '#246F72', '#1A4E50',
                      '#8F367C', '#1A5031', '#D86045', '#94422F', '#4853E0', '#7AE048', 
                      '#E07648', '#8B6A5B', '#4F5161', '#55614F', '#E0C548', '#8B835B',
                      '#614F5E', '#A9E15E', '#f6be65', '#E06648', '#486EE0', '#5B678B']
        
        # 生成更多颜色，确保至少有50种颜色
        from matplotlib import cm
        viridis = cm.get_cmap('viridis', 30)
        plasma = cm.get_cmap('plasma', 30)
        
        extended_colors = []
        for i in range(30):
            if i < 24:
                extended_colors.append(base_colors[i])
            
            if len(extended_colors) < 60:
                extended_colors.append(viridis(i/30))
            
            if len(extended_colors) < 60:
                extended_colors.append(plasma(i/30))
        
        self.colors = extended_colors
        
        # 存储各专家输出的字典
        self.expert_outputs = {}
        self.expert_features_collected = False
        
        # 钩子列表
        self.hooks = []
    
    def _register_hooks(self, model):
        """注册钩子函数来获取专家输出"""
        # 清除之前的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 为每个专家注册钩子
        try:
            for i, expert in enumerate(model.router.moe.experts):
                self.expert_outputs[i] = []
                
                def get_hook_fn(expert_idx):
                    def hook_fn(module, inputs, outputs):
                        # 只在评估阶段收集特征
                        if not module.training:
                            self.expert_outputs[expert_idx].append(outputs.detach().cpu())
                    return hook_fn
                
                # 为每个专家的前向传播注册钩子
                hook = expert.register_forward_hook(get_hook_fn(i))
                self.hooks.append(hook)
                
            self.expert_features_collected = False
        except Exception as e:
            print(f"注册专家钩子失败: {e}")
            # 清理已注册的钩子
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            raise e
            
    def _clear_expert_outputs(self):
        """清除之前收集的专家输出"""
        for key in self.expert_outputs:
            self.expert_outputs[key] = []
        self.expert_features_collected = False
    
    def _append_to_temp_file(self, file_path, new_data, is_label=False):
        """安全地将新数据追加到临时文件"""
        try:
            current_data = np.load(file_path)
            if is_label:
                # 标签是一维数组
                np.save(file_path, np.concatenate([current_data, new_data]))
            else:
                # 特征和门控概率是二维数组
                np.save(file_path, np.vstack([current_data, new_data]))
        except Exception as e:
            print(f"追加数据到临时文件 {file_path} 时出错: {e}")
            # 如果出错但文件不存在，则直接创建新文件
            if not os.path.exists(file_path):
                np.save(file_path, new_data)
    
    def extract_features(self, model, data_loader, device, num_samples=5000, extract_experts=False, verbose=True, batch_process=False):
        """
        从模型中提取特征
        
        Args:
            model: CM-MMoE模型
            data_loader: 数据加载器
            device: 设备(CPU/GPU)
            num_samples: 要提取的样本数
            extract_experts: 是否提取各专家的独立输出
            verbose: 是否显示进度条
            batch_process: 是否批处理模式(减少内存使用)
            
        Returns:
            features: 特征数组
            labels: 标签数组
            gate_probs: 门控概率数组（如果可用）
            expert_features: 专家特征字典（如果extract_experts=True）
        """
        model.eval()
        
        # 如果需要提取专家特征，注册钩子
        if extract_experts:
            try:
                self._register_hooks(model)
                self._clear_expert_outputs()
            except Exception as e:
                print(f"无法提取专家特征，将只提取MoE特征: {e}")
                extract_experts = False
        
        features_list = []
        labels_list = []
        gate_probs_list = []
        count = 0
        
        # 批处理模式下的临时文件
        temp_dir = None
        temp_files = {}  # 使用字典统一管理所有临时文件
        
        # 如果使用批处理模式，创建临时目录
        if batch_process:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="tsne_temp_")
            
            # 定义并初始化必要的临时文件
            temp_files['features'] = {
                'path': os.path.join(temp_dir, "temp_features.npy"),
                'initial_shape': (0, model.config["FUSION_IN"])
            }
            temp_files['labels'] = {
                'path': os.path.join(temp_dir, "temp_labels.npy"),
                'initial_shape': (0,)
            }
            
            # 只在需要门控概率时添加相应文件
            if extract_experts:
                temp_files['gate_probs'] = {
                    'path': os.path.join(temp_dir, "temp_gate_probs.npy"),
                    'initial_shape': (0, len(model.router.moe.experts))
                }
                
            # 初始化所有临时文件
            for key, file_info in temp_files.items():
                np.save(file_info['path'], np.zeros(file_info['initial_shape'], 
                       dtype=int if key == 'labels' else float))
            
            if verbose:
                print(f"批处理模式: 临时文件将保存在 {temp_dir}")
        
        try:
            with torch.no_grad():
                # 添加进度条
                data_iter = tqdm(data_loader, desc="提取特征", disable=not verbose) if verbose else data_loader
                
                for data in data_iter:
                    try:
                        # 获取数据 - 处理不同格式的数据加载器返回值
                        if len(data) == 4:  # 训练模式: (question, answer, image, mask)
                            question, answer, image, mask = data
                            type_str = None
                        elif len(data) == 6:  # 验证/测试模式: (question, answer, image, type_str, mask, image_original)
                            question, answer, image, type_str, mask, _ = data
                        else:
                            raise ValueError(f"不支持的数据格式: {len(data)}个元素")
                        
                        # 移动到设备
                        image = image.to(device)
                        question = question.to(device)
                        if mask is not None:
                            mask = mask.to(device)
                        
                        # 前向传播
                        pred, pred_mask, gate_prob, _ = model(image, question, mask)
                        
                        # 获取掩码预测
                        m0 = pred_mask[:, 0, :, :].unsqueeze(1) / 255
                        m1 = pred_mask[:, 1, :, :].unsqueeze(1) / 255
                        m2 = pred_mask[:, 2, :, :].unsqueeze(1) / 255
                        
                        # 获取图像特征
                        v = model.imgModel(pixel_values=image)["pooler_output"]
                        v = model.dropout(v)
                        v = model.lineV(v)
                        v = F.silu(v)
                        
                        # 获取文本特征
                        if model.textHead == "siglip_512":
                            q = model.textModel(input_ids=question["input_ids"])["pooler_output"]
                        elif model.textHead in model.clipList:
                            q = model.textModel(**question)["pooler_output"]
                            q = model.dropout(q)
                            q = model.lineQ(q)
                            q = F.silu(q)
                        else:
                            q = model.textModel(question)
                        
                        # 获取MoE特征
                        moe_features, _, _ = model.router(m0, m1, m2, v, q)
                        
                        # 收集特征、标签和门控概率
                        batch_features = moe_features.cpu().numpy()
                        batch_labels = answer.cpu().numpy()
                        batch_gate_probs = gate_prob.cpu().numpy() if gate_prob is not None else None
                        
                        if batch_process:
                            # 批处理模式: 将数据追加到临时文件
                            self._append_to_temp_file(temp_files['features']['path'], batch_features)
                            self._append_to_temp_file(temp_files['labels']['path'], batch_labels, is_label=True)
                            
                            # 只在有门控概率且extract_experts=True时处理
                            if batch_gate_probs is not None and 'gate_probs' in temp_files:
                                self._append_to_temp_file(temp_files['gate_probs']['path'], batch_gate_probs)
                            
                            # 清理内存
                            torch.cuda.empty_cache()
                        else:
                            # 标准模式: 将数据保存在内存中
                            features_list.append(batch_features)
                            labels_list.append(batch_labels)
                            if batch_gate_probs is not None:
                                gate_probs_list.append(batch_gate_probs)
                        
                        # 更新计数
                        count += len(answer)
                        if verbose:
                            data_iter.set_postfix(samples=count)
                        
                        if num_samples and count >= num_samples:
                            if verbose:
                                print(f"已达到指定样本数量: {num_samples}")
                            break
                    except Exception as e:
                        print(f"处理批次时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # 处理结果
            if batch_process:
                # 从临时文件加载数据
                features = np.load(temp_files['features']['path'])
                labels = np.load(temp_files['labels']['path'])
                gate_probs = None
                if 'gate_probs' in temp_files and os.path.exists(temp_files['gate_probs']['path']):
                    gate_probs = np.load(temp_files['gate_probs']['path'])
            else:
                # 从内存中合并数据
                if not features_list:
                    raise ValueError("没有成功提取特征")
                
                features = np.vstack(features_list)
                labels = np.concatenate(labels_list)
                gate_probs = np.vstack(gate_probs_list) if gate_probs_list else None
            
            # 如果有样本数限制，进行截断
            if num_samples:
                features = features[:num_samples]
                labels = labels[:num_samples]
                if gate_probs is not None:
                    gate_probs = gate_probs[:num_samples]
            
            # 处理专家输出
            expert_features = None
            if extract_experts and self.expert_outputs and any(len(outputs) > 0 for outputs in self.expert_outputs.values()):
                expert_features = {}
                for expert_idx, outputs in self.expert_outputs.items():
                    if outputs:
                        try:
                            expert_features[expert_idx] = torch.cat(outputs, dim=0).numpy()[:num_samples] if num_samples else torch.cat(outputs, dim=0).numpy()
                        except Exception as e:
                            print(f"处理专家 {expert_idx} 的特征时出错: {e}")
                
                self.expert_features_collected = True
            
            # 移除钩子
            if extract_experts:
                for hook in self.hooks:
                    hook.remove()
                self.hooks = []
                
            result = {
                'features': features,
                'labels': labels
            }
            
            if gate_probs is not None:
                result['gate_probs'] = gate_probs
                
            if expert_features is not None:
                result['expert_features'] = expert_features
            
            return result
        
        finally:
            # 清理临时文件
            if batch_process and temp_dir:
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    if verbose:
                        print(f"已清理临时目录: {temp_dir}")
                except Exception as e:
                    print(f"清理临时目录失败: {e}")
    
    def save_features(self, features, labels, epoch=None, gate_probs=None, expert_features=None):
        """保存特征到文件"""
        prefix = f"epoch_{epoch}_" if epoch is not None else ""
        save_dir = os.path.join(self.save_dir, f"epoch_{epoch}") if epoch is not None else self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存主要特征和标签
        feature_file = os.path.join(save_dir, f"{prefix}Feature42.txt")
        label_file = os.path.join(save_dir, f"{prefix}Answer42.txt")
        
        np.savetxt(feature_file, features)
        np.savetxt(label_file, labels, fmt='%d')
        
        print(f"特征已保存至: {feature_file}")
        print(f"标签已保存至: {label_file}")
        
        # 保存门控概率（如果有）
        if gate_probs is not None:
            gate_file = os.path.join(save_dir, f"{prefix}GateProbs.txt")
            np.savetxt(gate_file, gate_probs)
            print(f"门控概率已保存至: {gate_file}")
        
        # 保存专家特征（如果有）
        if expert_features is not None:
            expert_dir = os.path.join(save_dir, f"{prefix}experts")
            os.makedirs(expert_dir, exist_ok=True)
            
            for expert_idx, expert_feature in expert_features.items():
                expert_file = os.path.join(expert_dir, f"Expert{expert_idx}_Features.txt")
                np.savetxt(expert_file, expert_feature)
                print(f"专家 {expert_idx} 特征已保存至: {expert_file}")
        
        return feature_file, label_file
    
    def create_visualization(self, features, labels, save_path, title=""):
        """创建t-SNE可视化"""
        print(f"生成t-SNE可视化，处理{len(features)}个样本...")
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)//5), random_state=42, verbose=1)
        features_tsne = tsne.fit_transform(features)
        
        # 归一化到[-0.5, 0.5]
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        features_normalized = scaler.fit_transform(features_tsne)
        
        # 创建可视化
        plt.style.use("classic")
        plt.figure(figsize=(12, 12))  # 增大图像尺寸
        ax = plt.gca()
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        print(f"绘制{len(unique_labels)}个不同类别...")
        
        # 绘制散点图
        for i, label in enumerate(unique_labels):
            idx = labels == label
            if np.sum(idx) > 0:  # 确保该类有样本
                color_idx = int(label) % len(self.colors)  # 使用标签值作为颜色索引
                ax.scatter(
                    features_normalized[idx, 0], 
                    features_normalized[idx, 1], 
                    c=[self.colors[color_idx]], 
                    alpha=0.7, 
                    s=5,  # 增大点大小以提高可见性
                    label=f"Class {int(label)}"
                )
        
        # 设置坐标轴
        ax.set_xlim(-0.5, 0.5)  # 使用独立参数而非列表
        ax.set_ylim(-0.5, 0.5)  # 使用独立参数而非列表
        ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 使用正确导入的MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 使用正确导入的MultipleLocator
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置坐标轴标签
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        
        # 不添加标题
        plt.title("")
        
        # 为大量类别优化图例处理
        num_classes = len(unique_labels)
        if num_classes > 0:
            if num_classes <= 20:
                # 少于20个类别，正常显示图例
                plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            else:
                # 大量类别时，创建多列图例
                ncols = max(1, num_classes // 25 + 1)  # 每25个类别一列
                plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1), 
                           ncol=ncols, fontsize='xx-small')  # 减小字体大小
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"可视化已保存至: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
    
    def visualize_from_model(self, model, data_loader, device, epoch=None, num_samples=5000, extract_experts=False, batch_process=False):
        """从模型直接创建可视化"""
        # 提取特征
        result = self.extract_features(
            model, 
            data_loader, 
            device, 
            num_samples, 
            extract_experts, 
            batch_process=batch_process
        )
        
        features = result['features']
        labels = result['labels']
        gate_probs = result.get('gate_probs')
        expert_features = result.get('expert_features')
        
        # 保存特征
        self.save_features(features, labels, epoch, gate_probs, expert_features)
        
        # 生成可视化
        prefix = f"epoch_{epoch}_" if epoch is not None else ""
        save_dir = os.path.join(self.save_dir, f"epoch_{epoch}") if epoch is not None else self.save_dir
        save_path = os.path.join(save_dir, f"{prefix}tsne.png")
        
        # 创建可视化，不设置标题
        self.create_visualization(features, labels, save_path, title="")
        
        return save_path
    
    def visualize_from_files(self, feature_file, label_file, save_path=None):
        """从已保存的特征文件创建可视化"""
        # 读取特征和标签
        features = np.loadtxt(feature_file)
        labels = np.loadtxt(label_file, dtype=int)
        
        # 如果没有指定保存路径，生成一个
        if save_path is None:
            base_dir = os.path.dirname(feature_file)
            save_path = os.path.join(base_dir, "tsne_from_file.png")
        
        # 创建可视化
        self.create_visualization(features, labels, save_path)
        
        return save_path
        
    def visualize_experts_contribution(self, model, data_loader, device, save_path=None, num_samples=1000):
        """可视化专家贡献"""
        print("提取专家贡献数据...")
        model.eval()
        gate_probs_list = []
        labels_list = []
        count = 0
        
        # 提取专家门控概率
        with torch.no_grad():
            for data in tqdm(data_loader, desc="提取门控概率"):
                try:
                    # 获取数据 - 处理不同格式的数据加载器返回值
                    if len(data) == 4:  # 训练模式: (question, answer, image, mask)
                        question, answer, image, mask = data
                    elif len(data) == 6:  # 验证/测试模式: (question, answer, image, type_str, mask, image_original)
                        question, answer, image, type_str, mask, _ = data
                    else:
                        raise ValueError(f"不支持的数据格式: {len(data)}个元素")
                    
                    # 移动到设备
                    image = image.to(device)
                    question = question.to(device)
                    if mask is not None:
                        mask = mask.to(device)
                    
                    # 前向传播获取门控概率
                    _, _, gate_prob, _ = model(image, question, mask)
                    
                    # 收集门控概率和标签
                    gate_probs_list.append(gate_prob.cpu().numpy())
                    labels_list.append(answer.cpu().numpy())
                    
                    # 更新计数
                    count += len(answer)
                    if num_samples and count >= num_samples:
                        break
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not gate_probs_list:
            raise ValueError("没有成功提取门控概率")
        
        # 合并门控概率和标签
        gate_probs = np.vstack(gate_probs_list)
        labels = np.concatenate(labels_list)
        
        # 如果有样本数限制，进行截断
        if num_samples:
            gate_probs = gate_probs[:num_samples]
            labels = labels[:num_samples]
        
        # 计算每个类别对专家的平均使用情况
        num_experts = gate_probs.shape[1]
        unique_labels = np.unique(labels)
        expert_usage = np.zeros((len(unique_labels), num_experts))
        
        for i, label in enumerate(unique_labels):
            idx = labels == label
            if np.sum(idx) > 0:
                expert_usage[i] = np.mean(gate_probs[idx], axis=0)
        
        # 创建热力图
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        im = ax.imshow(expert_usage, cmap='viridis')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 设置坐标轴标签 - 保留刻度但不显示标签
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(num_experts))
        ax.set_yticks(np.arange(len(unique_labels)))
        ax.set_xticklabels([f'E{i}' for i in range(num_experts)])
        ax.set_yticklabels([f'C{label}' for label in unique_labels])
        
        # 移除标题
        plt.title('')
        plt.tight_layout()
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.save_dir, "expert_contribution.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"专家贡献可视化已保存至: {save_path}")
        
        plt.close()
        
        return save_path

    def visualize_experts_representation(self, model, data_loader, device, save_path=None, expert_names=None, num_samples=1000, perplexity=30, point_size=10, add_text_labels=False):
        """
        可视化不同专家在特征空间中的分布
        
        Args:
            model: CM-MMoE模型
            data_loader: 数据加载器
            device: 设备(CPU/GPU)
            save_path: 保存路径，如果为None则自动生成
            expert_names: 专家名称列表，如果为None则使用默认名称
            num_samples: 要处理的样本数量
            perplexity: t-SNE的perplexity参数
            point_size: 散点大小
            add_text_labels: 是否添加文本标签和底部说明文字，默认为False
            
        Returns:
            save_path: 保存的图像路径
        """
        print("提取专家特征表示...")
        model.eval()
        
        # 注册钩子来获取专家的中间特征
        self._register_hooks(model)
        self._clear_expert_outputs()
        
        # 处理少量样本来获取专家的特征表示
        count = 0
        with torch.no_grad():
            for data in tqdm(data_loader, desc="提取专家特征表示"):
                try:
                    # 获取数据 - 处理不同格式的数据加载器返回值
                    if len(data) == 4:  # 训练模式: (question, answer, image, mask)
                        question, answer, image, mask = data
                    elif len(data) == 6:  # 验证/测试模式: (question, answer, image, type_str, mask, image_original)
                        question, answer, image, type_str, mask, _ = data
                    else:
                        raise ValueError(f"不支持的数据格式: {len(data)}个元素")
                    
                    # 移动到设备
                    image = image.to(device)
                    question = question.to(device)
                    if mask is not None:
                        mask = mask.to(device)
                    
                    # 前向传播，钩子将自动收集专家输出
                    model(image, question, mask)
                    
                    # 更新计数
                    count += len(answer)
                    if count >= num_samples:
                        break
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 检查是否成功收集了专家特征
        if not self.expert_outputs or not any(len(outputs) > 0 for outputs in self.expert_outputs.values()):
            raise ValueError("未能收集到专家特征，请检查钩子是否正确注册")
        
        # 处理每个专家的特征
        expert_features = {}
        for expert_idx, outputs in self.expert_outputs.items():
            if outputs:
                try:
                    # 合并所有批次的输出
                    features = torch.cat(outputs, dim=0)
                    
                    # 转换为numpy数组并取前num_samples个样本
                    features = features.numpy()
                    if len(features) > num_samples:
                        features = features[:num_samples]
                    
                    # 如果特征是高维的，先通过PCA降维以加速t-SNE
                    if features.shape[1] > 50:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=50)
                        features = pca.fit_transform(features)
                    
                    expert_features[expert_idx] = features
                    print(f"专家 {expert_idx}: 提取 {len(features)} 个样本的特征，维度 {features.shape[1]}")
                except Exception as e:
                    print(f"处理专家 {expert_idx} 的特征时出错: {e}")
        
        # 移除钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 检查是否有足够的专家特征
        if not expert_features:
            raise ValueError("未能提取到任何专家特征")
        
        # 为每个专家单独应用t-SNE，并将结果合并
        from sklearn.manifold import TSNE
        
        all_features = []
        all_expert_ids = []
        
        print("应用t-SNE降维...")
        for expert_idx, features in expert_features.items():
            # 应用t-SNE降维
            tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features)//4), 
                         random_state=42, max_iter=1000, learning_rate='auto',
                         init='random', verbose=0)
            
            try:
                features_tsne = tsne.fit_transform(features)
                
                # 保存降维后的特征和对应的专家ID
                all_features.append(features_tsne)
                all_expert_ids.extend([expert_idx] * len(features_tsne))
            except Exception as e:
                print(f"专家 {expert_idx} 的t-SNE降维失败: {e}")
                continue
        
        # 将所有降维后的特征合并
        if not all_features:
            raise ValueError("所有专家的t-SNE降维均失败")
        
        all_features = np.vstack(all_features)
        all_expert_ids = np.array(all_expert_ids)
        
        # 归一化所有特征到[-0.5, 0.5]范围
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        all_features_normalized = scaler.fit_transform(all_features)
        
        # 创建可视化
        plt.style.use("classic")
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        
        # 准备专家名称
        num_experts = len(np.unique(all_expert_ids))
        if expert_names is None:
            # 默认专家名称
            default_names = [
                "Standard Expert", "Sobel Edge Expert", "Schmid Texture Expert",
                "Gabor Filter Expert", "Wavelet Expert", "Color Histogram Expert",
                "HOG Expert", "LBP Expert"
            ]
            
            # 如果默认名称不够，添加通用名称
            expert_names = []
            for i in range(num_experts):
                if i < len(default_names):
                    expert_names.append(default_names[i])
                else:
                    expert_names.append(f"Expert {i}")
        
        # 确保专家名称足够
        if len(expert_names) < num_experts:
            for i in range(len(expert_names), num_experts):
                expert_names.append(f"Expert {i}")
        
        # 定义鲜明的颜色，确保每个专家用不同颜色
        distinct_colors = [
            '#FF5733', '#33FF57', '#5733FF', '#FF33A1', '#33A1FF',
            '#A1FF33', '#FF5733', '#33FF57', '#5733FF', '#FF33A1'
        ]
        
        # 如果颜色不够，从matplotlib的colormap中生成更多
        if len(distinct_colors) < num_experts:
            from matplotlib import cm
            cmap = cm.get_cmap('tab10', num_experts)
            distinct_colors = [cmap(i) for i in range(num_experts)]
        
        # 计算每个专家的中心点
        expert_centers = {}
        for expert_idx in np.unique(all_expert_ids):
            idx = all_expert_ids == expert_idx
            expert_centers[expert_idx] = np.mean(all_features_normalized[idx], axis=0)
        
        # 绘制散点图，为每个专家使用不同颜色
        for i, expert_idx in enumerate(np.unique(all_expert_ids)):
            idx = all_expert_ids == expert_idx
            color = distinct_colors[i % len(distinct_colors)]
            ax.scatter(
                all_features_normalized[idx, 0],
                all_features_normalized[idx, 1],
                c=color,
                s=point_size,
                alpha=0.8,
                label=expert_names[i % len(expert_names)]
            )
            
            # 添加专家标签文本
            if add_text_labels:
                center = expert_centers[expert_idx]
                ax.text(
                    center[0], center[1],
                    expert_names[i % len(expert_names)],
                    fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
                )
        
        # 设置坐标轴
        ax.set_xlim(-0.5, 0.5)  # 使用独立参数而非列表
        ax.set_ylim(-0.5, 0.5)  # 使用独立参数而非列表
        ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 使用正确导入的MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 使用正确导入的MultipleLocator
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置坐标轴标签
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        
        # 不添加标题，或者根据需求设置
        plt.title("")
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        
        # 添加底部说明文字
        if add_text_labels:
            plt.figtext(
                0.5, 0.01,
                "Visualization shows how different experts in CM-MMoE specialize in different feature spaces.\nEach cluster represents an expert's learned representation.",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.save_dir, "expert_representation.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"专家表示可视化已保存至: {save_path}")
        
        plt.close()  # 关闭图形以释放内存
        
        return save_path


# 便捷函数，用于在训练过程中调用
def extract_and_save_features(model, data_loader, save_dir, device=None, num_samples=None, extract_experts=False, batch_process=False, prefix="", verbose=True):
    """
    提取并保存特征的便捷函数
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        save_dir: 保存目录
        device: 运行设备，如果为None则自动检测
        num_samples: 要处理的样本数量
        extract_experts: 是否提取专家输出
        batch_process: 是否使用批处理模式减少内存使用
        prefix: 文件名前缀
        verbose: 是否显示详细信息
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        visualizer = MMoEVisualizer(save_dir=save_dir)
        result = visualizer.extract_features(
            model, 
            data_loader, 
            device, 
            num_samples, 
            extract_experts, 
            verbose=verbose,
            batch_process=batch_process
        )
        
        visualizer.save_features(
            result['features'], 
            result['labels'], 
            None, 
            result.get('gate_probs'), 
            result.get('expert_features')
        )
        
        return result
    except Exception as e:
        print(f"提取和保存特征过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 在训练过程中调用的钩子函数
def save_features_hook(model, data_loader, epoch, save_dir, device=None, save_interval=5, num_samples=5000, extract_experts=False, batch_process=False, verbose=True):
    """
    用作训练回调的钩子函数
    
    Args:
        model: 当前模型
        data_loader: 数据加载器
        epoch: 当前epoch
        save_dir: 保存目录
        device: 运行设备，如果为None则自动检测
        save_interval: 保存间隔
        num_samples: 样本数量
        extract_experts: 是否提取专家特征
        batch_process: 是否使用批处理模式减少内存使用
        verbose: 是否显示详细信息
    """
    if epoch % save_interval == 0 or epoch == model.num_epochs - 1:
        try:
            if verbose:
                print(f"正在保存epoch {epoch}的特征...")
            
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            features_dir = os.path.join(save_dir, f"epoch_{epoch}")
            visualizer = MMoEVisualizer(save_dir=features_dir)
            
            # 1. 生成标准的t-SNE可视化
            visualizer.visualize_from_model(
                model=model,
                data_loader=data_loader,
                device=device,
                epoch=epoch,
                num_samples=num_samples,
                extract_experts=extract_experts,
                batch_process=batch_process
            )
            
            # 2. 生成专家贡献可视化（热图）
            expert_contribution_path = os.path.join(features_dir, f"epoch_{epoch}_expert_contribution.png")
            visualizer.visualize_experts_contribution(
                model=model,
                data_loader=data_loader,
                device=device,
                save_path=expert_contribution_path,
                num_samples=min(1000, num_samples)  # 使用较少样本以提高速度
            )
            
            # 3. 生成专家表示可视化（按专家着色的散点图）
            expert_representation_path = os.path.join(features_dir, f"epoch_{epoch}_expert_representation.png")
            visualizer.visualize_experts_representation(
                model=model,
                data_loader=data_loader,
                device=device,
                save_path=expert_representation_path,
                num_samples=min(1000, num_samples),  # 使用较少样本以提高速度
                add_text_labels=False  # 默认不显示文本标签和底部说明
            )
            
            if verbose:
                print(f"Epoch {epoch} 的特征和可视化已保存到 {features_dir}")
        except Exception as e:
            print(f"保存Epoch {epoch}特征时出错: {e}")
            import traceback
            traceback.print_exc() 