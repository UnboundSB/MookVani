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
        use_rnn=False, # <-- The magic toggle!
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
        
        # 1. RNN ENCODER (Toggleable!)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
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
        
        # 3. CNN Classifier
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
        
        # 1. (Optional) RNN Encoder
        if self.use_rnn:
            lengths_cpu = lengths.cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            
        # 2. Transformer
        out_trans = self.pos_encoder(x)
        out_trans = self.transformer(out_trans, src_key_padding_mask=pad_mask)
        
        # 3. CNN Classifier
        out_cnn = out_trans.transpose(1, 2)
        logits = self.cnn(out_cnn)
        logits = logits.transpose(1, 2)
        
        logits = logits.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        
        return F.log_softmax(logits, dim=-1), lengths
