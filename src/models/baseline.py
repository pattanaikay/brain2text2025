import torch
import torch.nn as nn

class BrainToTextModel(nn.Module):
    def __init__(self, num_features=512, num_classes=28, hidden_size=512):
        super(BrainToTextModel, self).init()
        
        # 1. Spatial-Temporal Convolutional Layers
        # We treat the 512 channels as the input 'channels' for the 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # 2. Sequence Modeling (BiGRU)
        # input_size is hidden_size from CNN, output is hidden_size * 2 (bidirectional)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=2, 
                          bidirectional=True, batch_first=True, dropout=0.3)
        
        # 3. Output layer (logits for CTC)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> (Batch, 512, Time) for Conv1d
        x = x.transpose(1, 2)
        x = self.cnn(x)
        
        # (Batch, hidden_size, Time) -> (Batch, Time, hidden_size) for RNN
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        
        # Output Logits
        logits = self.fc(x)
        return logits # (Batch, Time, Classes)