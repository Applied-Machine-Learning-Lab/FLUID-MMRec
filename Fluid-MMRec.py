import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import numpy as np

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, layer_norm_eps):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.w_1 = nn.Linear(d_model, d_ff*2)
        self.w_2 = nn.Linear(d_ff*2, d_model)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(1, 1, d_model) * 0.1)
        
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return residual + self.layer_scale * x

class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        mamba_config = config['mamba']
        
        self.num_layers = mamba_config['num_layers']
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=mamba_config['hidden_dim'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand']
            ) for _ in range(self.num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(mamba_config['hidden_dim'], eps=mamba_config['norm_eps'])
            for _ in range(self.num_layers)
        ])
        
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, mamba_config['hidden_dim']) * 0.1)
            for _ in range(self.num_layers)
        ])
        
    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.norms[i](x)
            x = self.mamba_layers[i](x)
            x = self.dropout(x)
            x = residual + self.layer_scales[i] * x
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_attention_heads']
        self.hidden_size = config['multimodal']['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        self.image_projection = nn.Sequential(
            nn.Linear(config['image']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout_prob'])
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(config['text']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout_prob'])
        )
        
        self.image_to_text_heads = nn.ModuleList([
            nn.MultiheadAttention(self.head_dim, 1, dropout=config['dropout_prob']) 
            for _ in range(self.num_heads)
        ])
        
        self.text_to_image_heads = nn.ModuleList([
            nn.MultiheadAttention(self.head_dim, 1, dropout=config['dropout_prob'])
            for _ in range(self.num_heads)
        ])
        
        self.head_norms = nn.ModuleList([
            nn.LayerNorm(self.head_dim * 2)
            for _ in range(self.num_heads)
        ])
        
        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, image_features, text_features, attention_mask=None):
        batch_size, seq_len = image_features.size(0), image_features.size(1)
        
        image_proj = self.image_projection(image_features)  # [B, L, H]
        text_proj = self.text_projection(text_features)     # [B, L, H]
        
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        image_heads = image_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        text_heads = text_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        image_heads = image_heads.permute(0, 2, 1, 3)  # [B, H, L, D]
        text_heads = text_heads.permute(0, 2, 1, 3)    # [B, H, L, D]
        
        if attention_mask is not None:
            key_padding_mask = (~attention_mask).to(image_features.device).bool()
        else:
            key_padding_mask = None
        
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            curr_image = image_heads[:, head_idx]  # [B, L, D]
            curr_text = text_heads[:, head_idx]    # [B, L, D]
            
            curr_image = curr_image * self.scaling
            curr_text = curr_text * self.scaling
            
            curr_image_t = curr_image.transpose(0, 1)  # [L, B, D]
            curr_text_t = curr_text.transpose(0, 1)    # [L, B, D]
            
            img2text_output, _ = self.image_to_text_heads[head_idx](
                query=curr_image_t,
                key=curr_text_t,
                value=curr_text_t,
                key_padding_mask=key_padding_mask
            )
            
            text2img_output, _ = self.text_to_image_heads[head_idx](
                query=curr_text_t,
                key=curr_image_t,
                value=curr_image_t,
                key_padding_mask=key_padding_mask
            )
            
            img2text = img2text_output.transpose(0, 1)
            text2img = text2img_output.transpose(0, 1)
            
            img2text = curr_image + img2text
            text2img = curr_text + text2img
            
            combined = torch.cat([img2text, text2img], dim=-1)  # [B, L, 2D]
            normalized = self.head_norms[head_idx](combined)
            
            head_outputs.append(normalized)
        
        return torch.stack(head_outputs, dim=1)

class ExpertRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config['expert']['num_experts']
        self.hidden_size = config['router']['hidden_size']
        
        head_dim = config['multimodal']['hidden_size'] // config['num_attention_heads']
        input_dim = head_dim * 2
        
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim), 
            nn.Linear(input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config['router']['dropout']),
            nn.LayerNorm(self.hidden_size), 
            nn.Linear(self.hidden_size, self.num_experts)
        )
        
        self.temperature = 0.1
        self.balance_coefficient = 0.001 
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x, training=False):
        batch_size, num_heads, seq_len, dim = x.size()
        
        x = F.normalize(x.reshape(-1, dim), dim=-1).view(batch_size, num_heads, seq_len, dim)
        
        x_reshaped = x.view(batch_size * num_heads, seq_len, dim)
        logits = self.router(x_reshaped)
        
        log_gates = F.log_softmax(logits / self.temperature, dim=-1)
        
        if training:
            noise = torch.rand_like(log_gates) * 0.05
            log_gates = log_gates + noise
            
            gates = F.gumbel_softmax(log_gates, tau=self.temperature, hard=True, dim=-1)
        else:
            gates = torch.exp(log_gates) 
        
        gates = gates.view(batch_size, num_heads, seq_len, -1)
        
        gates = torch.clamp(gates, min=1e-6, max=1.0)
        
        if training:
            expert_usage = gates.mean(dim=(0, 1, 2))
            expert_usage = torch.clamp(expert_usage, min=1e-6)
            balance_loss = -torch.sum(expert_usage * torch.log(expert_usage))
            balance_loss = self.balance_coefficient * torch.clamp(balance_loss, max=1.0)
            return gates, balance_loss
        
        return gates, None

class SteinKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kernel_type = config['bottleneck']['kernel_type']
        self.adaptive_bandwidth = config['bottleneck']['adaptive_bandwidth']
        self.min_bandwidth = config['bottleneck']['min_bandwidth']
        self.max_bandwidth = config['bottleneck']['max_bandwidth']
        self.bandwidth_factor = config['bottleneck']['bandwidth_factor']
        self.register_buffer('bandwidth', torch.tensor(1.0))
        self.chunk_size = 128 

    def update_bandwidth(self, x, y):
        with torch.no_grad():
            sample_size = min(512, x.size(0))
            x_sample = x[:sample_size]
            y_sample = y[:sample_size]
            
            diff = x_sample.unsqueeze(1) - y_sample.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            
            bandwidth_sq = torch.median(dist_sq.view(-1))
            bandwidth = torch.sqrt(bandwidth_sq / 2.0) * self.bandwidth_factor
            
            self.bandwidth = torch.clamp(
                bandwidth,
                min=self.min_bandwidth,
                max=self.max_bandwidth
            )
        
    def rbf_kernel(self, x, y):
        self.update_bandwidth(x, y)
        
        batch_size = x.size(0)
        kernel_matrix = torch.zeros(batch_size, batch_size, device=x.device)
        
        for i in range(0, batch_size, self.chunk_size):
            end_i = min(i + self.chunk_size, batch_size)
            chunk_x = x[i:end_i]
            
            for j in range(0, batch_size, self.chunk_size):
                end_j = min(j + self.chunk_size, batch_size)
                chunk_y = y[j:end_j]
                
                diff = chunk_x.unsqueeze(1) - chunk_y.unsqueeze(0)
                dist_sq = torch.sum(diff ** 2, dim=-1)
                
                if self.bandwidth is None:
                    bandwidth_sq = torch.median(dist_sq.view(-1))
                    bandwidth = torch.sqrt(bandwidth_sq / 2.0) * self.bandwidth_factor
                    self.bandwidth = torch.clamp(bandwidth, self.min_bandwidth, self.max_bandwidth)
                
                kernel_matrix[i:end_i, j:end_j] = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))
        
        return kernel_matrix

    def score_kernel(self, x, score_x, y, score_y):
        """
        Args:
            x: [B, D]
            score_x: [B, D]
            y: [B, D]
            score_y: [B, D]
        Returns:
            kernel_matrix: [B, B]
        """
        self.update_bandwidth(x, y)
        
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [B, B, D]
        dist_sq = torch.sum(diff ** 2, dim=-1)   # [B, B]
        kernel_mat = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))  # [B, B]
        
        term1 = kernel_mat.unsqueeze(-1) * score_x.unsqueeze(1)  # [B, B, D]
        term1 = torch.sum(term1 * score_y.unsqueeze(0), dim=-1)  # [B, B]
        
        term2 = torch.sum(diff * score_y.unsqueeze(0), dim=-1)  # [B, B]
        term2 = term2 * kernel_mat / (self.bandwidth ** 2)
        
        term3 = torch.sum(diff ** 2, dim=-1) * kernel_mat / (self.bandwidth ** 4)
        term3 = term3 - kernel_mat * x.size(-1) / (self.bandwidth ** 2)
        
        return term1 + term2 + term3  # [B, B]


class MultiViewEntropyBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['multimodal']['hidden_size']
        self.output_dim = config['bottleneck']['dim']
        self.beta = config['bottleneck']['beta']
        
        self.image_encoder = nn.Sequential(
            nn.Linear(config['image']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_dim * 2)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(config['text']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_dim * 2)
        )
        
        score_hidden = self.output_dim * 2
        self.image_score = nn.Sequential(
            nn.Linear(self.output_dim, score_hidden),
            nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        
        self.text_score = nn.Sequential(
            nn.Linear(self.output_dim, score_hidden),
            nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        
        self.stein_kernel = SteinKernel(config)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def compute_kl_loss(self, mu1, logvar1, mu2, logvar2, mask=None):
        kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)  # [B, L]
        kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)  # [B, L]
        
        if mask is not None:
            sequence_lengths = mask.sum(dim=1).long() - 1  # [B]
            batch_indices = torch.arange(mask.size(0), device=mask.device)
            kl1 = kl1[batch_indices, sequence_lengths]  # [B]
            kl2 = kl2[batch_indices, sequence_lengths]  # [B]
            
        return (kl1.mean() + kl2.mean()) / 2
        
    def forward(self, image_features, text_features, attention_mask=None):
        """
        Args:
            image_features: [B, L, D1]
            text_features: [B, L, D1]
            attention_mask: [B, L]
        """
        batch_size = image_features.size(0)
        
        image_params = self.image_encoder(image_features)  # [B, L, D*2]
        text_params = self.text_encoder(text_features)     # [B, L, D*2]
        
        image_mu, image_logvar = torch.chunk(image_params, 2, dim=-1)
        text_mu, text_logvar = torch.chunk(text_params, 2, dim=-1)

        image_z = self.reparameterize(image_mu, image_logvar)
        text_z = self.reparameterize(text_mu, text_logvar)
        
        image_z = F.normalize(image_z, p=2, dim=-1)  # [B, L, D]
        text_z = F.normalize(text_z, p=2, dim=-1)    # [B, L, D]
        
        image_score = self.image_score(image_z)  # [B, L, D]
        text_score = self.text_score(text_z)     # [B, L, D]
        
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1).long() - 1  # [B]
            batch_indices = torch.arange(batch_size, device=image_z.device)
            
            image_z_last = image_z[batch_indices, sequence_lengths]      # [B, D]
            text_z_last = text_z[batch_indices, sequence_lengths]        # [B, D]
            image_score_last = image_score[batch_indices, sequence_lengths]  # [B, D]
            text_score_last = text_score[batch_indices, sequence_lengths]    # [B, D]
        else:
            image_z_last = image_z[:, -1]        # [B, D]
            text_z_last = text_z[:, -1]          # [B, D]
            image_score_last = image_score[:, -1]  # [B, D]
            text_score_last = text_score[:, -1]    # [B, D]
        
        kernel_matrix = self.stein_kernel.score_kernel(
            image_z_last, image_score_last,
            text_z_last, text_score_last
        )  # [B, B]
        
        alignment_loss = torch.mean(kernel_matrix)
        
        kl_loss = self.compute_kl_loss(
            image_mu, image_logvar,
            text_mu, text_logvar,
            attention_mask
        )
        
        total_loss = kl_loss + self.beta * alignment_loss
        
        return {
            'image_repr': image_z,  # [B, L, D]
            'text_repr': text_z,    # [B, L, D]
            'alignment_loss': alignment_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

class Fliud_MMRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(Fliud_MMRec, self).__init__(config, dataset)
        
        self.dataset = dataset
        self.config = config
        
        self.id_embedding_dim = config['id_embedding_dim']
        self.modal_hidden_size = config['multimodal']['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.loss_type = config['loss_type']

        self.num_heads = config['num_attention_heads']
        self.head_dim = self.modal_hidden_size // self.num_heads
        
        self.image_features = torch.from_numpy(
            np.load(config['feature_paths']['image_feature_path'])
        ).float().to(config['device'])
        self.text_features = torch.from_numpy(
            np.load(config['feature_paths']['text_feature_path'])
        ).float().to(config['device'])
        
        self.item_embedding = nn.Embedding(self.n_items, self.id_embedding_dim, padding_idx=0)
        
        self.image_projection = nn.Linear(config['image']['feature_dim'], 
                                        config['image']['projection_dim'])
        self.text_projection = nn.Linear(config['text']['feature_dim'], 
                                       config['text']['projection_dim'])
        
        self.id_mamba = MambaLayer(config)  
        self.fusion_mamba = MambaLayer(config)  
        
        self.cross_attention = MultiHeadCrossAttention(config)
        self.router = ExpertRouter(config)
        self.stein_mveb = MultiViewEntropyBottleneck(config)
        self.bottleneck_weight = config['bottleneck']['weight']

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 2, self.id_embedding_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_prob'])
            )
            for _ in range(config['expert']['num_experts'])
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * config['num_attention_heads'], 
                     self.id_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.id_embedding_dim),
            nn.Dropout(config['multimodal']['fusion_dropout'])
        )
        
        self.final_fusion = self.fusion_mamba
        self.final_norm = nn.LayerNorm(self.id_embedding_dim)
        
        self.loss_fct = BPRLoss() if self.loss_type == 'BPR' else nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_multimodal_features(self, item_seq):
        """Get multimodal features for the items in the sequence"""
        # Convert item IDs to tokens
        iid_series = self.dataset.id2token(self.dataset.iid_field, item_seq.cpu())
        
        # Create mask for valid items (not padding)
        mask = (iid_series != '[PAD]')
        
        # Convert mask to correct device and type
        attention_mask = mask.to(self.device).bool()
        
        # Prepare indices array
        batch_size, seq_len = item_seq.size()
        indices = torch.zeros_like(item_seq)
        
        # Safely handle index conversion
        valid_indices = []
        valid_positions = []
        
        # Process each position
        for b in range(batch_size):
            for s in range(seq_len):
                if mask[b, s].item():
                    item_id = iid_series[b, s]
                    if item_id != '[PAD]':
                        item_idx = int(item_id) - 1  # -1 for 0-indexing
                        if 0 <= item_idx < len(self.image_features):
                            indices[b, s] = item_idx
                            valid_indices.append(item_idx)
                            valid_positions.append((b, s))
                        else:
                            attention_mask[b, s] = False
        
        # Initialize feature tensors
        image_features = torch.zeros(batch_size, seq_len, self.image_features.size(1), 
                                   device=self.device, dtype=self.image_features.dtype)
        text_features = torch.zeros(batch_size, seq_len, self.text_features.size(1), 
                                  device=self.device, dtype=self.text_features.dtype)
        
        # Safely fill features
        for i, (b, s) in enumerate(valid_positions):
            idx = valid_indices[i]
            if idx >= 0 and idx < len(self.image_features):
                image_features[b, s] = self.image_features[idx]
                text_features[b, s] = self.text_features[idx]
        
        return image_features, text_features, attention_mask
    
    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        id_seq_output = self.id_mamba(item_emb)
        
        image_features, text_features, mask = self.get_multimodal_features(item_seq)
        
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)
        
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)

        mveb_output = self.stein_mveb(image_proj, text_proj, mask)
        image_aligned = mveb_output['image_repr']
        text_aligned = mveb_output['text_repr']
        
        self.mveb_loss = mveb_output['total_loss']
        
        head_features = self.cross_attention(image_aligned, text_aligned, mask)
        
        gates, _ = self.router(head_features)
        
        expert_outputs = []
        batch_size = item_seq.size(0)
        seq_len = item_seq.size(1)
        
        for head_idx in range(self.num_heads):
            head_output = head_features[:, head_idx]
            head_gates = gates[:, head_idx]
            
            head_expert_output = torch.zeros(
                batch_size, seq_len, self.id_embedding_dim,
                device=item_seq.device
            )
            
            for expert_idx, expert in enumerate(self.experts):
                expert_out = expert(head_output)
                expert_out = F.normalize(expert_out, p=2, dim=-1)
                weight = head_gates[:, :, expert_idx].unsqueeze(-1)
                head_expert_output += expert_out * weight
            
            expert_outputs.append(head_expert_output)
        
        combined_output = torch.cat(expert_outputs, dim=-1)
        
        modal_features = self.feature_fusion(combined_output)
        modal_features = F.normalize(modal_features, p=2, dim=-1)
        
        final_features = torch.cat([id_seq_output, modal_features], dim=-1)
        fused_features = self.final_fusion(final_features)
        final_output = self.final_norm(fused_features)
        
        seq_output = self.gather_indexes(final_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)

        mveb_weight = self.bottleneck_weight
        total_loss = rec_loss + mveb_weight * self.mveb_loss
        
        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores