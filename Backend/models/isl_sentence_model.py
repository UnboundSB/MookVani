import torch
import torch.nn as nn
import torch.nn.functional as F
from .isl_conformer import ISL_Conformer

class ISL_Sentence_Model(nn.Module):
    """
    A wrapper model for Sentence-level Sign Language Translation.
    Uses a pre-trained ISL_Conformer as a feature extractor, and appends a 
    Bidirectional LSTM to temporally smooth/stack frame-level predictions before
    final classification and CTC alignment.
    """
    def __init__(
        self,
        num_classes=100,
        d_model=256,
        lstm_layers=1,
        dropout=0.5,
        freeze_base=True
    ):
        super().__init__()
        
        # 1. Base Feature Extractor (The Conformer)
        # Note: We initialize the base model with its original num_classes (2000 for words)
        # just in case we load strict weights, but we will ignore its final classifier.
        self.base_model = ISL_Conformer(num_classes=2000, d_model=d_model)
        
        # 2. Temporal Smoothing Layer (BiLSTM)
        # We use d_model // 2 for hidden_size so the output is concatenated to exactly d_model.
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 3. New Sentence-Level Classifier (+1 for CTC Blank)
        self.classifier = nn.Linear(d_model, num_classes + 1)
        
        self.freeze_base = freeze_base

    def load_base_weights(self, checkpoint_path):
        """Loads pre-trained word-level weights into the base Conformer model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle DataParallel wrapped states
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if "classifier" not in name:
                new_state_dict[name] = v
            
        # Load weights strictly into base_model (ignoring classifier mismatches if any)
        self.base_model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded base Conformer weights from {checkpoint_path}")
        
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Frozen base Conformer weights (only training BiLSTM + Classifier).")

    def forward(self, x, lengths):
        # 1. Extract dense features from Conformer
        with torch.set_grad_enabled(not self.freeze_base):
            x_fused, lengths_pooled, pad_mask_bt_pooled = self.base_model.extract_features(x, lengths)
            
        # x_fused is (B, T_pooled, d_model)
        
        # 2. Pack the sequence for the LSTM to ignore padding
        # CPU conversion required for pack_padded_sequence lengths
        lengths_cpu = lengths_pooled.cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_fused, 
            lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 3. Pass through BiLSTM
        packed_out, _ = self.bilstm(packed_x)
        
        # 4. Unpack sequence
        out_bilstm, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, 
            batch_first=True, 
            total_length=x_fused.size(1)
        )
        
        # 5. Final Classification
        logits = self.classifier(out_bilstm)
        
        # Apply padding mask to zero-out padding logits just in case
        logits = logits.masked_fill(pad_mask_bt_pooled.unsqueeze(-1), 0.0)
        
        return F.log_softmax(logits, dim=-1), lengths_pooled
