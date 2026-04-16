import pandas as pd
from src.utils.decoders import greedy_decoder

def generate_submission(model, test_loader, tokenizer, device):
    model.eval()
    results = []

    with torch.no_grad():
        for neural_inputs, test_ids in test_loader:
            logits = model(neural_inputs.to(device))
            
            # Loop through the batch
            for i in range(logits.size(0)):
                pred_text = greedy_decoder(logits[i], tokenizer)
                results.append({
                    "id": test_ids[i],
                    "sentence": pred_text
                })
    
    # T-503: Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("submissions/submission.csv", index=False)
    print("Submission saved to submissions/submission.csv")