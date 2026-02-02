from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict
import omegaconf


def masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling that excludes padding tokens using attention_mask.
    
    Args:
        last_hidden_state: (batch, seq_len, hidden)
        attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
        
    Returns:
        pooled: (batch, hidden)
    """
    # (batch, seq_len, 1)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-6)  # avoid division by zero
    return masked.sum(dim=1) / denom

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()
        """
        Initialize encoder model for classification
        
        Args:
            model_config: OmegaConf config containing model name, hidden_size, num_labels
        """
        self.model_name = model_config.model.name
        self.hidden_size = model_config.model.hidden_size
        self.num_labels = model_config.model.num_labels
        
        # Load pretrained model
        if 'modernbert' in self.model_name.lower():
            # ModernBERT doesn't support add_pooling_layer argument
            self.encoder = AutoModel.from_pretrained(self.model_name)
        else:
            # For BERT and other models
            self.encoder = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, 
                input_ids : torch.Tensor, 
                attention_mask : torch.Tensor, 
                token_type_ids : torch.Tensor = None,
                label : torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        
        Outputs :
            outputs : dict containing {
                'logits': (batch_size, num_labels),
                'loss': (1,) if label is provided else None
            }
        """
        # Prepare model inputs
        model_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add token_type_ids if available (BERT models)
        if token_type_ids is not None:
            model_input['token_type_ids'] = token_type_ids
        
        # Get encoder output
        encoder_output = self.encoder(**model_input)
        
        # Pool the output (masked mean pooling over tokens)
        last_hidden_state = encoder_output['last_hidden_state']  # (batch_size, seq_len, hidden_size)
        pooled_output = masked_mean_pooling(last_hidden_state, attention_mask)  # (batch_size, hidden_size)
        
        # Get logits
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        
        # Calculate loss if labels are provided
        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)
        
        return {
            'logits': logits,
            'loss': loss
        }
