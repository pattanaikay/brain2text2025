import torch
import os
import sys
from pathlib import Path

# Add parent directory to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchview import draw_graph
from src.models.baseline import BITModel
from src.models.encoder import BIT_Transformer
from src.models.projector import MLPProjector

def visualize_components():
    """Visualize individual custom components first for clarity."""
    print("Generating Neural Encoder visualization...")
    encoder = BIT_Transformer(session_ids=["session_1", "session_2"])
    # Input: (Batch, Time, Channels) -> (1, 500, 512)
    encoder_graph = draw_graph(encoder, input_size=(1, 500, 512), device='cpu', graph_name="NeuralEncoder")
    encoder_graph.visual_graph.render("outputs/neural_encoder_arch", format="png")
    
    print("Generating MLP Projector visualization...")
    projector = MLPProjector()
    # Input: (Batch, Time_patch, 384) -> (1, 100, 384)
    projector_graph = draw_graph(projector, input_size=(1, 100, 384), device='cpu', graph_name="MLPProjector")
    projector_graph.visual_graph.render("outputs/projector_arch", format="png")

def visualize_full_model():
    """Visualize the full BIT pipeline."""
    print("Generating Full BIT Model visualization (this may take a moment due to LLM size)...")
    # We use a smaller mock LLM or set depth to avoid a massive graph
    # For visualization, we don't need to load the actual 1.5B weights if we just want the structure
    try:
        model = BITModel(quantize=False) # Load on CPU for visualization
        
        # We set depth=2 to see: Model -> Encoder, Projector, LLM
        # Without expanding the thousands of layers inside the LLM
        full_graph = draw_graph(
            model, 
            input_data=torch.randn(1, 500, 512),
            device='cpu',
            depth=2, 
            graph_name="BIT_Full_Pipeline",
            expand_nested=True
        )
        full_graph.visual_graph.render("outputs/bit_full_architecture", format="png")
        print("Success: Full model graph saved to outputs/bit_full_architecture.png")
    except Exception as e:
        print(f"Note: Full model visualization skipped or failed (likely due to LLM memory/loading): {e}")
        print("Falling back to component-level visualization...")
        visualize_components()

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    
    # Check if graphviz is installed in the system
    # torchview requires the graphviz system binary, not just the python package
    import subprocess
    try:
        subprocess.run(["dot", "-V"], check=True, capture_output=True)
        print("Graphviz found. Generating component visualizations...\n")
        visualize_components()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "="*60)
        print("WARNING: System 'graphviz' (dot) not found.")
        print("="*60)
        print("To generate visualizations, install graphviz:")
        print("  Windows (using choco): choco install graphviz")
        print("  Ubuntu/Debian: sudo apt install graphviz")
        print("  macOS: brew install graphviz")
        print("\nSkipping visualization generation for now.")
        print("="*60 + "\n")
