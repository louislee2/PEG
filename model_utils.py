# peg_pipeline/model_utils.py
import math
import torch
import torch.nn as nn

# PositionalEncoding, NextActivityTransformer, build_user_model, ModelWrapper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NextActivityTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=6, d_ff=512,
                 dropout=0.1, max_seq_len=100, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size + 1)

        # try to capture attention via hooks
        self.attention_weights = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        def hook_fn(module, input, output):
            # output may include attn weights for some implementations; best-effort
            if len(output) > 1 and output[1] is not None:
                self.attention_weights.append(output[1].detach())
        for name, module in self.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        self.attention_weights = []
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0,1)).transpose(0,1)
        x = self.dropout(x)
        src_key_padding_mask = (input_ids == self.pad_idx) if attention_mask is None else (attention_mask == 0)
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_projection(encoded)
        result = {"logits": logits}
        if output_attentions:
            result["attentions"] = tuple(self.attention_weights) if self.attention_weights else None
        return result

def build_user_model(activity_to_id: dict, cfg: dict=None, device=None):
    cfg = cfg or {}
    d_model = cfg.get("d_model", 128)
    n_heads = cfg.get("n_heads", 8)
    n_layers = cfg.get("n_layers", 6)
    d_ff = cfg.get("d_ff", 512)
    dropout = cfg.get("dropout", 0.1)
    max_seq_len = cfg.get("max_seq_len", 100)
    pad_idx = cfg.get("pad_id", 0)

    vocab_size = max(activity_to_id.values())
    model = NextActivityTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_idx=pad_idx
    )

    # init weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
    model.apply(init_weights)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()
    return model, pad_idx

class ModelWrapper:
    def __init__(self, model: nn.Module, activity_to_id: dict, id_to_activity: dict, pad_id: int = 0, device=None):
        self.model = model
        self.activity_to_id = activity_to_id
        self.id_to_activity = id_to_activity
        self.pad_id = pad_id
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

    def _encode_batch(self, batch_prefixes, max_len=None):
        if max_len is None:
            max_len = max(len(p) for p in batch_prefixes)
        B = len(batch_prefixes)
        input_ids = torch.full((B, max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        for i, p in enumerate(batch_prefixes):
            L = len(p)
            input_ids[i, :L] = torch.tensor(p, dtype=torch.long, device=self.device)
            attn_mask[i, :L] = 1
        return input_ids, attn_mask

    def predict_proba(self, batch_prefixes, batch_size=64):
        import numpy as np
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(batch_prefixes), batch_size):
                batch = batch_prefixes[i:i+batch_size]
                input_ids, attn_mask = self._encode_batch(batch)
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=False)
                logits = outputs.get("logits", None) if isinstance(outputs, dict) else None
                if logits is None:
                    raise RuntimeError("模型 forward 没有返回 logits")
                if logits.dim() == 3:
                    last_pos = (attn_mask.sum(dim=1) - 1).long()
                    probs_batch = []
                    for b in range(logits.size(0)):
                        pos = int(last_pos[b].item())
                        logit = logits[b, pos, :]
                        probs = torch.softmax(logit, dim=-1)
                        probs_batch.append(probs.cpu().numpy())
                    probs_batch = np.stack(probs_batch, axis=0)
                elif logits.dim() == 2:
                    probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
                else:
                    raise RuntimeError(f"Unsupported logits dim: {logits.dim()}")
                all_probs.append(probs_batch)
        return np.vstack(all_probs)

    def get_attention_matrices(self, prefix):
        """
        返回 (num_layers, num_heads, L, L) 或 (num_heads, L, L) numpy 的注意力矩阵（若可得）
        """
        input_ids, attn_mask = self._encode_batch([prefix])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
            if isinstance(outputs, dict) and "attentions" in outputs and outputs["attentions"] is not None:
                attentions = outputs["attentions"]
                try:
                    import torch as _torch
                    att_stack = _torch.stack([a.squeeze(0) for a in attentions], dim=0)
                    return att_stack.cpu().numpy()
                except Exception:
                    return None
            if hasattr(self.model, 'attention_weights') and self.model.attention_weights:
                try:
                    import torch as _torch
                    att_stack = _torch.stack([a.squeeze(0) for a in self.model.attention_weights], dim=0)
                    return att_stack.cpu().numpy()
                except Exception:
                    return None
        return None

    def predict_with_embedding_zero(self, prefix, mask_positions):
        """
        把指定位置 embedding 置 0，返回预测概率向量 (V,)
        """
        embedding_layer = self.model.embedding
        def hook_fn(module, input, output):
            output_modified = output.clone()
            for pos in mask_positions:
                if pos < output_modified.size(1):
                    output_modified[0, pos, :] = 0.0
            return output_modified
        handle = embedding_layer.register_forward_hook(hook_fn)
        try:
            input_ids, attn_mask = self._encode_batch([prefix])
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=False)
                logits = outputs.get("logits", None) if isinstance(outputs, dict) else None
                if logits is None:
                    raise RuntimeError("模型 forward 没有返回 logits")
                if logits.dim() == 3:
                    pos = int(attn_mask.sum(dim=1).item()) - 1
                    logit = logits[0, pos, :]
                else:
                    logit = logits[0]
                probs = torch.softmax(logit, dim=-1).cpu().numpy()
        finally:
            handle.remove()
        return probs

    def predict_with_input_mask(self, prefix, mask_positions):
        """
        通过把指定位置替换为 pad_id 来做 input-mask（真实删除该位置的影响）。
        返回预测概率向量 (V,)
        """
        # make a copy of prefix with masked positions set to pad_id (0)
        masked = list(prefix[:])
        for pos in mask_positions:
            if pos < len(masked):
                masked[pos] = self.pad_id  # 替换为 PAD id
        input_ids, attn_mask = self._encode_batch([masked])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=False)
            logits = outputs.get("logits", None) if isinstance(outputs, dict) else None
            if logits is None:
                raise RuntimeError("模型 forward 没有返回 logits")
            if logits.dim() == 3:
                pos = int(attn_mask.sum(dim=1).item()) - 1
                logit = logits[0, pos, :]
            else:
                logit = logits[0]
            probs = torch.softmax(logit, dim=-1).cpu().numpy()
        return probs

    def predict_with_attention_mask(self, prefix, mask_positions):
        """
        将指定位置的注意力矩阵行/列置零，返回预测概率向量 (V,)
        """
        input_ids, attn_mask = self._encode_batch([prefix])
        self.model.attention_weights = []  # 清空注意力权重

        def attention_hook(module, input, output):
            if len(output) > 1 and output[1] is not None:
                attn_weights = output[1].clone()  # [batch, num_heads, seq_len, seq_len]
                for pos in mask_positions:
                    if pos < attn_weights.size(-1):
                        attn_weights[:, :, :, pos] = 0.0  # 屏蔽目标列（被关注位置）
                        attn_weights[:, :, pos, :] = 0.0  # 屏蔽目标行（关注其他位置）
                return (output[0], attn_weights)
            return output

        # 注册hook
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                handles.append(module.register_forward_hook(attention_hook))

        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
                logits = outputs.get("logits", None)
                if logits is None:
                    raise RuntimeError("模型 forward 没有返回 logits")
                pos = int(attn_mask.sum(dim=1).item()) - 1
                logit = logits[0, pos, :] if logits.dim() == 3 else logits[0]
                probs = torch.softmax(logit, dim=-1).cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        return probs

