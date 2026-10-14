import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ISLSentenceReconformer(nn.Module):
    """
    User-Designed Architecture: (Optional) RNN -> Transformer -> CNN
    By setting use_rnn=False, we can pretrain the Transformer + CNN natively.
    Then later, we can set use_rnn=True to slide the BiGRU in front and expand it!
    """
    def __init__(
        self, 
        input_dim=163, 
        num_classes=263, 
        d_model=256, 
        lstm_layers=2, 
        trans_layers=4, 
        num_heads=8, 
        dropout=0.3,
        use_rnn=True, # Default to True: Data enters RNN -> Transformer -> TCNN
        freeze_base=False
    ):
        super().__init__()
        
        self.use_rnn = use_rnn
        self.freeze_base = freeze_base
        
        # Pre-Dense projection to expand raw frames to d_model
        self.input_dense = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 1. RNN ENCODER: Bi-directional GRU
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.rnn_norm = nn.LayerNorm(d_model)
        
        # 2. TRANSFORMER (Global Attention)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, trans_layers)
        
        # 3. TCNN Classifier (Temporal CNN)
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(d_model // 2, num_classes + 1, kernel_size=1)
        )

    def load_base_weights(self, checkpoint_path):
        """Safely loads weights. Allows for dropping the RNN dynamically."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            
            # STRIP THE HEAD: Prevent size mismatch crashes if pretraining on 500 classes and fine-tuning on 263!
            if "cnn.8.weight" in name or "cnn.8.bias" in name:
                continue
                
            new_state_dict[name] = v
            
        # strict=False allows us to dynamically add/remove the RNN and ignore the stripped head!
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}. (Head stripped successfully)")
        
        if self.freeze_base:
            for name, param in self.named_parameters():
                if "transformer" in name or "input_dense" in name:
                    param.requires_grad = False
            print("Frozen Transformer and Input Dense! Only training BiGRU + CNN.")

    def forward(self, x, lengths):
        B, T, _ = x.shape
        arange = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pad_mask = arange >= lengths.unsqueeze(1)
        
        # 0. Initial Dense Expansion
        x = self.input_dense(x)
        
        # 1. RNN Encoder: Bi-directional temporal context
        if self.use_rnn:
            lengths_cpu = lengths.cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            x = self.rnn_norm(x + rnn_out)
            
        # 2. Transformer: Global cross-frame attention
        out_trans = self.pos_encoder(x)
        out_trans = self.transformer(out_trans, src_key_padding_mask=pad_mask)
        
        # 3. TCNN Classifier: Local multi-scale feature refinement
        out_cnn = out_trans.transpose(1, 2)
        logits = self.cnn(out_cnn)
        logits = logits.transpose(1, 2)
        
        logits = logits.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        
        return F.log_softmax(logits, dim=-1), lengths

    def forward_word_logits(self, x, lengths):
        """
        Extracts temporal frames through RNN -> Transformer -> TCNN and applies
        masked temporal average pooling in logit space.
        Returns:
            logits_pooled: (B, num_classes + 1) un-normalized pooled logits
        """
        B, T, _ = x.shape
        arange = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pad_mask = arange >= lengths.unsqueeze(1)
        
        # 0. Initial Dense Expansion
        x = self.input_dense(x)
        
        # 1. RNN Encoder: Bi-directional temporal context
        if self.use_rnn:
            lengths_cpu = lengths.cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            x = self.rnn_norm(x + rnn_out)
            
        # 2. Transformer: Global cross-frame attention
        out_trans = self.pos_encoder(x)
        out_trans = self.transformer(out_trans, src_key_padding_mask=pad_mask)
        
        # 3. TCNN Classifier: Local multi-scale feature refinement
        out_cnn = out_trans.transpose(1, 2)
        logits = self.cnn(out_cnn).transpose(1, 2) # (B, T, num_classes + 1)
        
        # Mask out padding frames and average across valid frames
        valid_mask = (~pad_mask).unsqueeze(-1).float()
        logits_masked = logits * valid_mask
        logits_pooled = logits_masked.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1).float()
        return logits_pooled

    def predict_word_probs(self, x, lengths):
        """
        Length-normalized probability pooling across sequence frames.
        Returns:
            probs_pooled: (B, num_classes + 1) normalized probabilities summing to 1.0
            log_probs_pooled: (B, num_classes + 1) log-probabilities for loss computation
        """
        logits_pooled = self.forward_word_logits(x, lengths)
        probs_pooled = F.softmax(logits_pooled, dim=-1)
        log_probs_pooled = F.log_softmax(logits_pooled, dim=-1)
        return probs_pooled, log_probs_pooled
