# Zhang Paper Discussion: BIT Encoder, Projector, E2E Training, Losses, and QLoRA

This note summarizes the discussion around the Brain-to-Text Integration Transformer (BIT) style architecture inspired by Zhang et al. (2025), the attached technical specification, and the local code files:

- `brain2text-modeltraining/src/models/encoder.py`
- `brain2text-modeltraining/src/models/projector.py`
- `brain2text-modeltraining/src/models/bit_e2e.py`
- `brain2text-modeltraining/src/preprocessing/dataloader.py`
- `brain2text-modeltraining/scripts/train_ctc.py`

The goal is to explain the system from the top down, then connect the high-level concepts to the specific code modules.

## 1. Top-Down System View

The core idea is to translate neural activity into text using an end-to-end neural system.

At the broadest level:

```text
raw neural activity
-> neural encoder
-> projected LLM embeddings
-> prompt-wrapped LLM
-> generated sentence
```

Traditional speech BCI systems often use a cascaded pipeline:

```text
brain activity -> phonemes -> language model -> sentence
```

The BIT-style system tries to make this more integrated:

```text
brain activity -> neural representations -> LLM -> sentence
```

This allows the neural encoder, the projector, and parts of the LLM adaptation to be trained together, rather than treating phoneme decoding and language modeling as completely separate steps.

## 2. Architecture Phases

The rough architecture image describes four phases.

### Phase 1: SSL Pretraining

The encoder sees neural data without needing text labels. Some neural patches are masked, and the model learns to reconstruct the missing patches.

Conceptually:

```text
neural sequence
-> hide some time patches
-> predict the hidden neural patches
```

This teaches the encoder general neural structure: what normal neural activity patterns look like over time.

In `encoder.py`, this is supported by the `mask_token` and the `mask_patches` argument:

```python
if mask_patches is not None:
    mask_token = self.mask_token.expand_as(x).to(x.dtype)
    x = torch.where(mask_patches.unsqueeze(-1), mask_token, x)
```

The actual reconstruction head and MSE pretraining loop live outside the three core model files.

### Phase 2: Phoneme Finetuning

The encoder is trained to make its time-token outputs align with phoneme sequences.

Conceptually:

```text
neural activity -> encoder tokens -> phoneme classifier -> CTC loss
```

This teaches the encoder that its internal time representation should be useful for speech sounds, not just generic neural reconstruction.

### Phase 3: Sentence Finetuning

The encoder and projector are connected to an LLM. Projected neural tokens are inserted into a prompt, and the LLM is trained to generate the correct text.

Conceptually:

```text
neural activity
-> encoder
-> projector
-> LLM embedding prefix
-> target sentence
```

The model learns from paired examples:

```text
neural recording X
corresponding sentence Y
```

### Phase 4: Inference

At inference time, unseen neural activity is passed through the encoder and projector, then the LLM generates text autoregressively.

```text
unseen neural data
-> encoder
-> projector
-> prompt + neural embeddings
-> LLM.generate(...)
-> sentence
```

## 3. What the Encoder Does

The encoder learns "what pattern of brain activity is happening over time."

More concretely, it converts raw neural recordings:

```text
[B, T, 512]
```

into contextual neural tokens:

```text
[B, T_patch, 384]
```

Where:

- `B` is batch size.
- `T` is the number of neural time bins.
- `512` is the number of neural channels.
- `T_patch` is the compressed number of time patches.
- `384` is the encoder embedding dimension.

The encoder does this in several stages.

### 3.1 Session-Specific Read-In

Different recording sessions can have slightly different neural statistics due to probe drift, channel noise, or session-specific variation.

In `encoder.py`, the encoder optionally creates one read-in layer per session:

```python
self.read_in[safe_sid] = nn.Linear(input_dim, input_dim)
```

During the forward pass:

```python
x = layer(x)
```

This lets the model learn a session-specific correction or channel mixing step.

Conceptually:

```text
raw 512-channel activity
-> session-adjusted 512-channel activity
```

### 3.2 Time Patching

Neural recordings arrive as many small time bins. The encoder groups nearby bins into patches.

In `encoder.py`:

```python
pad_len = (self.patch_size - (time_steps % self.patch_size)) % self.patch_size
x = x.view(batch_size, new_time_steps // self.patch_size, self.patch_size * channels)
```

If `patch_size=4`:

```text
4 time bins * 512 channels = 2048 patch features
```

If `patch_size=5`:

```text
5 time bins * 512 channels = 2560 patch features
```

The attached diagram/spec often refer to 5 bins as 100 ms patches. The current code default is `patch_size=4`, so training scripts should be checked to confirm the intended value.

Patching reduces the sequence length, making transformer attention cheaper and giving each token a small local neural time window.

### 3.3 Patch Embedding

Each large patch is compressed into a 384-dimensional token:

```python
x = self.patch_ln1(x)
x = self.patch_embedding(x)
x = self.patch_ln2(x)
```

Conceptually:

```text
short neural window
-> compact neural token
```

This is where the model starts learning local neural features: combinations of channel activity over a short time window that are useful for later speech decoding.

### 3.4 Transformer Temporal Modeling

The transformer stack lets neural patches attend to one another over time:

```python
for layer in self.layers:
    x = layer(x, cos, sin, key_padding_mask=key_padding_mask)
```

This matters because speech is temporal. A local neural activation may only be meaningful in context.

Instead of asking:

```text
What happened in this one patch?
```

the transformer asks:

```text
What happened in this patch, given the patches before and after it?
```

Self-attention helps the model learn neural trajectories, such as:

```text
activation pattern A
-> followed by activation pattern B
-> followed by activation pattern C
```

The RoPE positional embeddings help the attention mechanism understand temporal order:

```python
q = apply_rotary_pos_emb(q, cos, sin)
k = apply_rotary_pos_emb(k, cos, sin)
```

## 4. Why the Transformer Learns Speech-Relevant Temporal Patterns

The transformer does not automatically know speech. It learns speech-relevant temporal patterns because the training data contains paired examples:

```text
neural activity recording
corresponding phonemes or sentence text
```

There are two related things being learned.

### 4.1 Temporal Neural Structure

The encoder learns how neural activity evolves over time:

```text
patch 1 -> patch 2 -> patch 3 -> ...
```

SSL reconstruction can help here, even without text labels.

### 4.2 Neural-to-Speech Mapping

The model learns which neural time patterns correspond to speech labels because it is trained on paired neural-speech examples.

During CTC phoneme finetuning:

```text
neural activity -> phoneme sequence
```

During E2E sentence finetuning:

```text
neural activity -> sentence text
```

So the clean mental model is:

```text
The transformer learns relationships across time inside the neural recording.
The losses tell it which relationships matter for speech.
```

Without labels, the encoder can learn general neural dynamics. To learn that a neural pattern corresponds to `/k/`, `/ae/`, or the word "cat", it needs paired neural-speech examples.

## 5. What the Projector Does

The projector is the bridge between the encoder and the LLM.

The encoder outputs 384-dimensional neural tokens:

```text
[B, T_patch, 384]
```

But the LLM expects embeddings in its own hidden size, such as 1536 dimensions for a 1.5B-scale Qwen/Aero-style model.

`projector.py` defines:

```python
self.mlp = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
    nn.LayerNorm(output_dim)
)
```

Conceptually:

```text
neural token space
-> LLM embedding space
```

The projector does not generate text. It translates the encoder's neural representations into vectors that can be inserted into the LLM as if they were input embeddings.

## 6. What `bit_e2e.py` Does

`bit_e2e.py` defines the full end-to-end model.

It combines:

- the neural encoder
- the MLP projector
- a tokenizer
- a quantized LLM
- LoRA adapters
- a CTC auxiliary head
- an InfoNCE-style contrastive loss
- generation logic

The forward path is:

```text
neural_data
-> neural_encoder
-> projector
-> projected neural embeddings
-> concatenate with prompt embeddings
-> append target text embeddings during training
-> LLM
-> losses
```

The prompt looks like:

```text
<neural_activity>
[projected neural embeddings]
</neural_activity>
decode the above neural activity into an English sentence:
```

The key detail is that the model feeds neural information into the LLM through `inputs_embeds`, not ordinary token IDs.

That allows non-text neural vectors to act as a soft prefix/context for the LLM.

## 7. Phoneme Finetuning and CTC

Phoneme finetuning forces encoder outputs to align with phoneme structure through CTC loss.

In `train_ctc.py`, a temporary phoneme model is defined:

```python
class CTCPhonemeModel(nn.Module):
    def __init__(self, encoder, num_phonemes=42):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_phonemes)

    def forward(self, x, session_id=None):
        encoded = self.encoder(x, session_id=session_id)
        return self.head(encoded)
```

This maps:

```text
neural data
[B, T, 512]

encoder output
[B, T_patch, 384]

phoneme logits
[B, T_patch, 42]
```

The 42 classes are:

```text
0 = CTC blank
1-41 = real phoneme classes
```

The dataloader shifts phoneme labels by `+1`:

```python
phonemes = phonemes + 1
```

This reserves class `0` for the CTC blank token.

### 7.1 Why CTC Is Needed

We know the phoneme sequence for an utterance, but we do not know the exact neural timestep where each phoneme begins and ends.

For example:

```text
Target phonemes:
K AE T
```

The encoder may produce many timestep predictions:

```text
blank blank K K blank AE AE blank T T blank
```

CTC collapses this into:

```text
K AE T
```

It does that by:

1. Collapsing adjacent repeated tokens.
2. Removing blank tokens.

So CTC asks:

```text
Is there some monotonic alignment through time that can produce the target phoneme sequence?
```

It does not need exact frame-level phoneme labels.

### 7.2 CTC Training in Code

In `train_ctc.py`:

```python
logits = model(neural_data, session_id=session_id)
logits = logits.permute(1, 0, 2)
log_probs = nn.functional.log_softmax(logits, dim=2)

input_lengths = (neural_lengths + model.encoder.patch_size - 1) // model.encoder.patch_size

loss = ctc_loss_fn(log_probs, labels, input_lengths, target_lengths)
```

PyTorch CTC expects:

```text
log_probs:      [T_patch, B, num_classes]
labels:         phoneme targets
input_lengths:  real neural patch lengths
target_lengths: real phoneme sequence lengths
```

The loss rewards the model when the predicted timestep distributions assign high probability to paths that collapse into the correct phoneme sequence.

### 7.3 How Phoneme Structure Is Checked

There are two checks.

First, the dataloader checks CTC feasibility:

```python
if phoneme_len > 0 and patched_len < phoneme_len:
    continue
```

CTC cannot align a target sequence that is longer than the input timestep sequence.

Second, validation computes phoneme error rate (PER). In `train_ctc.py`, predictions are greedily decoded:

```python
preds = logits.argmax(dim=-1)
```

Then the code:

1. Collapses adjacent duplicates.
2. Removes blank class `0`.
3. Compares the decoded phoneme sequence to the ground truth.

`calculate_per` uses `jiwer.wer` over space-separated phoneme IDs:

```python
def calculate_per(predictions, targets):
    return jiwer.wer(list(targets), list(predictions))
```

Low CTC loss means the model assigns probability to valid alignments.

Low PER means the decoded phoneme sequence actually matches the target sequence well.

## 8. Losses Used in the System

The system uses multiple losses because it is trying to learn multiple things:

```text
CTC:       make encoder tokens phoneme-aware
CE:        make the LLM generate the correct sentence
InfoNCE:   align neural and text embeddings globally
```

### 8.1 CTC Loss

CTC is used for phoneme alignment.

It handles:

```text
many neural timesteps
shorter phoneme sequence
unknown exact alignment
```

The model emits phoneme probabilities at every neural patch, and CTC checks whether those probabilities can collapse into the true phoneme sequence.

This forces the encoder to organize its time tokens in a speech-like order.

### 8.2 Cross-Entropy Loss

Cross-entropy is the standard LLM next-token prediction loss.

In `bit_e2e.py`:

```python
outputs = self.llm(
    inputs_embeds=full_embeds,
    attention_mask=attention_mask,
    labels=full_labels,
)
ce_loss = outputs.loss
```

The sequence given to the LLM is:

```text
prompt start embeddings
projected neural embeddings
prompt end embeddings
target sentence embeddings
```

The labels mask the prefix with `-100`, so the LLM is only punished for mistakes on the target sentence tokens.

Conceptually:

```text
Given this neural prefix, predict the correct next text token.
```

### 8.3 Contrastive Loss and InfoNCE

"Contrastive loss" is a broad family of losses that pull matching pairs together and push non-matching pairs apart.

In this code, the contrastive loss is InfoNCE-style.

In `bit_e2e.py`:

```python
neural_embeds = F.normalize(neural_embeds, p=2, dim=-1)
text_embeds = F.normalize(text_embeds, p=2, dim=-1)
logits = torch.matmul(neural_embeds, text_embeds.t()) / tau
labels = torch.arange(batch_size, device=neural_embeds.device)

loss_n = F.cross_entropy(logits, labels)
loss_t = F.cross_entropy(logits.t(), labels)
return (loss_n + loss_t) / 2
```

For a batch of four examples:

```text
neural_1 should match text_1
neural_2 should match text_2
neural_3 should match text_3
neural_4 should match text_4
```

The similarity matrix should have high values on the diagonal:

```text
              text_1  text_2  text_3  text_4
neural_1       high    low     low     low
neural_2       low     high    low     low
neural_3       low     low     high    low
neural_4       low     low     low     high
```

The off-diagonal pairs are treated as negatives.

In E2E training, the neural side is usually a mean-pooled projected neural embedding:

```python
neural_pooled = projected_embeds.mean(dim=1)
```

The text side is a masked mean-pool over label embeddings:

```python
text_pooled = (label_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
```

This tells the model:

```text
The whole neural recording should land near the embedding of its matching sentence.
```

### 8.4 Final E2E Loss

In `bit_e2e.py`, the total E2E loss is:

```python
loss = ce_loss + ctc_weight * ctc_loss + contrastive_loss
```

So the model is trained with three simultaneous pressures:

```text
CE:      produce the right sentence
CTC:     keep encoder outputs phoneme-aware
InfoNCE: align neural and text representations
```

## 9. QLoRA in LLMs

QLoRA means:

```text
Quantized Low-Rank Adaptation
```

It combines:

```text
Quantization: store the base LLM in fewer bits
LoRA:         train small adapter matrices instead of full LLM weights
```

### 9.1 Why QLoRA Is Useful

Full finetuning an LLM is expensive because the model has many parameters.

QLoRA makes training cheaper by:

1. Loading the pretrained base LLM in 4-bit precision.
2. Freezing most base model weights.
3. Training small LoRA adapter weights.

This lets the model adapt to a new task without updating every LLM parameter.

### 9.2 Quantization

In `bit_e2e.py`:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

This means:

```text
Base LLM weights are stored in 4-bit NF4 format.
Computation uses bfloat16.
Double quantization compresses quantization metadata further.
```

Then:

```python
self.llm = prepare_model_for_kbit_training(self.llm)
```

prepares the quantized model for adapter training.

### 9.3 LoRA

A normal linear layer computes:

```text
y = W x
```

Where `W` is a large pretrained weight matrix.

LoRA keeps `W` frozen and adds a trainable low-rank update:

```text
y = W x + LoRA_update(x)
```

The update is factorized into two smaller matrices:

```text
LoRA_update(x) = B(Ax) * scale
```

Instead of training a huge matrix:

```text
W: d_out x d_in
```

LoRA trains:

```text
A: r x d_in
B: d_out x r
```

Where `r` is much smaller than `d_in` or `d_out`.

In the local code:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.2,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

This adapts:

- attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- feed-forward projections: `gate_proj`, `up_proj`, `down_proj`

These are important internal LLM layers that control attention and representation transformation.

### 9.4 How QLoRA Works Here

In this BIT implementation:

```text
encoder/projector produce neural embeddings
neural embeddings are inserted into the LLM prompt
LLM must generate the correct sentence
```

During training, gradients can update:

- the neural encoder
- the projector
- LoRA adapter weights
- the CTC head
- the contrastive temperature

The base LLM is mostly frozen and quantized.

So QLoRA teaches the LLM:

```text
How should I use these projected neural embeddings as context for text generation?
```

It does not teach English from scratch. The LLM already knows language. QLoRA teaches it to condition on a new kind of prefix: neural activity embeddings.

## 10. Important Implementation Notes

There are a few important distinctions between the paper/spec/diagram and the current code.

### 10.1 LLM Choice

The diagram/spec discuss Aero-1-Audio-style integration, but the current `bit_e2e.py` default is:

```python
llm_name="Qwen/Qwen2.5-1.5B-Instruct"
```

So the current implementation is using a text LLM by default, not necessarily Aero-1-Audio.

### 10.2 Patch Size

The architecture image and spec often describe:

```text
5 bins of 20 ms = 100 ms patch
```

But the current code default is:

```python
patch_size=4
```

The training scripts should be checked to ensure SSL, CTC, and E2E all use the same intended patch size.

### 10.3 SSL Support vs Full SSL Implementation

`encoder.py` supports masked patches through `mask_token`, but the full reconstruction head and MSE loss are not in `encoder.py`.

### 10.4 CTC as Separate Stage and Auxiliary E2E Loss

There is a dedicated CTC finetuning stage in `train_ctc.py`.

`bit_e2e.py` also includes a CTC auxiliary loss during E2E:

```python
loss = ce_loss + ctc_weight * ctc_loss + contrastive_loss
```

This helps preserve phoneme-aware structure while training the sentence-level model.

## 11. Compact Mental Model

The full system can be remembered as:

```text
The read-in layer aligns neural channels across sessions.
The patch embedding turns short neural windows into compact tokens.
The transformer learns how those neural tokens evolve over time.
CTC teaches those tokens to follow phoneme order.
The projector maps neural tokens into LLM embedding space.
InfoNCE pulls matching neural/text examples together.
Cross-entropy teaches the LLM to generate the correct sentence.
QLoRA lets the LLM adapt to neural prefixes without full finetuning.
```

Or even shorter:

```text
Encoder:  learns speech-relevant neural time patterns.
Projector: translates neural tokens into LLM-compatible embeddings.
LLM:       turns those embeddings into English text.
Losses:    tell each part what "useful" means.
QLoRA:     adapts the LLM cheaply to this new neural input modality.
```
