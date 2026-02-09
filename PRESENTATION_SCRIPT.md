# Qwen-Image Workshop Presentation Script
## AMD MI300X GPU-Powered Image Generation Workshop

**Target Audience:** 3rd & 4th Year Computer Science/Engineering Students
**Duration:** 60-90 minutes
**Presenter Guide:** Cell-by-cell walkthrough

---

## PRE-PRESENTATION SETUP (5 minutes before start)

### What to Have Ready:
- [ ] Jupyter notebook open and kernel ready
- [ ] AMD ROCm GPU monitoring tool running (amd-smi in separate terminal)
- [ ] Browser tab with AMD Instinct MI300X specs
- [ ] This script for reference

### Opening Hook (First 2 minutes):
"Good [morning/afternoon] everyone! Today, we're going to do something pretty incredible. We're going to work with an **86 BILLION parameter** AI model that can understand text in multiple languages—including Korean and Kannada—generate photorealistic images, edit them, merge multiple subjects, and even train custom adaptations.

And here's the kicker: we're doing ALL of this on a **single AMD Instinct MI300X GPU**. On a consumer GPU, this would be completely impossible. Let's dive in!"

---

## PART 1: INTRODUCTION & CONTEXT

### Cell 0-2: Title and Prerequisites (2 minutes)

**[Show cells but don't execute]**

**Script:**
"This tutorial was developed by ModelScope and Tongyi Lab from Alibaba Group. Let's quickly talk about what we're working with today:

**Hardware - AMD Instinct MI300X:**
- 192 GB of HBM3 memory (For context, a high-end consumer GPU has 24GB)
- This massive memory is why we can load multiple huge models simultaneously
- Built for enterprise AI workloads

**Software Stack:**
- ROCm: AMD's open-source GPU computing platform (think CUDA but open-source)
- DiffSynth-Studio: A unified framework for diffusion models
- PyTorch with ROCm support

**The Models:**
- Qwen-Image: 86B parameter image generation model
- Qwen-Image-Edit: Specialized for image editing
- Qwen-Image-Edit-2509: Multi-image editing and composition

**Quick poll: How many of you have:**
- Worked with image generation models before? (Stable Diffusion, DALL-E?)
- Used PyTorch for deep learning?
- Heard of LoRA (Low-Rank Adaptation)?

Great! We'll cover concepts from the ground up, so everyone can follow along."

---

## PART 2: ENVIRONMENT SETUP

### Cell 3-4: Hardware Verification (3 minutes)

**[Execute Cell 4 - amd-smi]**

**Script:**
"First, let's verify our GPU. I'm going to run `amd-smi`—AMD's System Management Interface.

**For Beginners:**
Think of this like checking your computer's Task Manager, but specifically for AMD GPUs.

**For Advanced Students:**
This tool shows us:
- GPU utilization and temperature
- Memory usage (VRAM)
- Power consumption
- Clock speeds
- ROCm version

[Run the command and point out key metrics on screen]

See that memory? **192 GB**. That's not a typo. This is why we can do things today that are impossible on consumer hardware.

**Real-world context:**
Training large language models or loading multiple models for production inference requires this level of memory. Companies like OpenAI, Anthropic, and Meta use these types of GPUs."

---

### Cell 5-6: Installing DiffSynth-Studio (4 minutes)

**[Execute Cell 6]**

**Script:**
"Now we're installing DiffSynth-Studio from source. Let me explain what's happening here:

**For Beginners:**
We're downloading a framework that makes it easy to work with diffusion models—the type of AI that generates images from text.

**For Intermediate Students:**
Notice we're:
1. Cloning from GitHub
2. Checking out a specific commit (afd101f3...) for reproducibility
3. Creating a custom requirements file for AMD ROCm
4. Installing with `-e .` (editable install)

**For Advanced Students:**
Pay attention to this requirements file:

```python
--index-url https://download.pytorch.org/whl/rocm6.4
```

We're using PyTorch wheels compiled specifically for ROCm 6.4. This is crucial:
- ROCm-specific kernels for optimal performance
- HIP (Heterogeneous Interface for Portability) backend instead of CUDA
- Binary compatibility with AMD GPU architecture

**Key point:** Notice we add the package to `sys.path` immediately. This is a notebook-specific trick to avoid kernel restarts.

[Wait for installation to complete]

**Discussion question:** Why do you think we pin a specific git commit instead of using the latest version?"

**Expected answers:** Reproducibility, stability, avoiding breaking changes

---

## PART 3: MODEL LOADING & INFERENCE

### Cell 7-8: Loading the Base Model (5 minutes)

**[Execute Cell 8 - This is critical]**

**Script:**
"This is where the magic happens. We're loading an **86 billion parameter model** into GPU memory.

**For Beginners:**
Parameters are like the 'knowledge' in an AI model. More parameters generally mean more capability. For comparison:
- GPT-2: 1.5 billion parameters
- GPT-3: 175 billion parameters
- This model: 86 billion parameters (but specialized for images)

**For Intermediate Students:**
Let's break down this code:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
```
We're forcing **offline mode**. Why? Because we've pre-downloaded all models to save time and bandwidth. In a workshop with 50+ users, we don't want everyone downloading 80GB+ of models!

```python
ModelConfig(model_id="Qwen/Qwen-Image",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
```

We're using **safetensors** format—a safer alternative to pickle that prevents code injection attacks.

**For Advanced Students:**
Notice the architecture:
- **Transformer**: The core diffusion model (the heavy lifter)
- **Text Encoder**: Converts your prompt to embeddings
- **VAE** (Variational Autoencoder): Compresses images to/from latent space

This is a **Latent Diffusion Model** architecture:
1. Text → Embeddings (Text Encoder)
2. Noise + Embeddings → Latent Image (Transformer)
3. Latent Image → Pixel Image (VAE Decoder)

The `torch.bfloat16` dtype gives us:
- 50% memory reduction vs float32
- Better numerical stability than float16
- Maintained accuracy for transformer models

[Monitor GPU memory while loading]

**See the memory usage jump?** We've just loaded tens of billions of parameters into VRAM.

**Real-world insight:** Companies use model compression techniques like quantization, pruning, and distillation to make these models smaller. But with MI300X's 192GB, we can load the full-precision model!"

---

### Cell 9-10: First Image Generation (5 minutes)

**[Execute Cell 10]**

**Script:**
"Let's generate our first image with a simple prompt: 'a portrait of a beautiful Asian woman'

**For Beginners:**
What's happening:
1. Your text is converted to numbers (embeddings)
2. The model starts with random noise
3. Over 40 steps, it gradually 'denoises' the image
4. We get a photorealistic result

**For Intermediate Students:**
Key parameters:
- `seed=0`: Deterministic randomness (same seed = same output)
- `num_inference_steps=40`: More steps = better quality but slower

The diffusion process works backward from noise to image. Each step:
```
x_t-1 = denoising_function(x_t, text_embedding, timestep)
```

**For Advanced Students:**
This is implementing DDPM (Denoising Diffusion Probabilistic Models):
- Forward process: q(x_t | x_t-1) adds Gaussian noise
- Reverse process: p_θ(x_t-1 | x_t) learns to denoise
- The transformer predicts the noise at each timestep

The 40 inference steps use a noise scheduler (likely DDIM or DPM-Solver) to sample from the learned distribution.

[Show generated image]

**Discussion:** Notice anything about the quality? It might lack fine details. That's intentional—we'll improve it next!

**Performance note:** Watch the generation speed. On consumer GPUs, this might take 30-60 seconds. On MI300X? Much faster due to memory bandwidth and compute power."

---

## PART 4: ENHANCEMENT WITH LoRA

### Cell 11-12: Loading LoRA (4 minutes)

**[Execute Cell 12]**

**Script:**
"Now we're going to enhance our model using **LoRA** - Low-Rank Adaptation.

**For Beginners:**
Think of LoRA like adding a specialized skill to an already trained model. Instead of retraining the entire 86B parameter model (which would take weeks and cost $100,000+), we train a small 'adapter' that modifies the model's behavior.

**For Intermediate Students:**
LoRA works by:
1. Freezing the original model weights (W)
2. Adding trainable low-rank matrices: W + B×A
3. B and A are much smaller than W

Example:
- Original weight matrix: 4096×4096 = 16M parameters
- LoRA with rank=32: (4096×32) + (32×4096) = 262K parameters
- **99% reduction in trainable parameters!**

**For Advanced Students:**
The mathematical formulation:

```
W' = W + ΔW
ΔW = B × A
```

Where:
- W ∈ ℝ^(d×k) (frozen pretrained weights)
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k) (trainable, r << d, k)
- r is the rank (hyperparameter, typically 4-64)

This leverages the hypothesis that adaptation has a low "intrinsic rank"—the required changes lie in a low-dimensional subspace.

Benefits:
- Memory efficient (only store A and B)
- Can swap LoRAs for different tasks
- No inference latency if merged
- Multiple LoRAs can be combined

The `enable_lora_magic()` and `hotload=True` enable dynamic LoRA swapping without reloading the base model.

**Industry application:** Companies like Stability AI use this to offer hundreds of different 'styles' without storing hundreds of full models."

---

### Cell 13-14: Enhanced Generation (3 minutes)

**[Execute Cell 14]**

**Script:**
"Same prompt, but now with the ArtAug-v1 LoRA loaded. Let's see the difference!

[Show the image]

**Point out improvements:**
- Finer details in facial features
- Better skin texture
- Improved lighting and composition
- More artistic quality

**For Advanced Students:**
This LoRA was trained on high-quality artistic images. It's essentially learned a mapping:
- Input: Qwen-Image's latent representations
- Output: Enhanced latent representations with artistic qualities

The LoRA weights modify the attention mechanisms to prioritize certain visual features.

**Key insight:** We didn't change the model's core knowledge, just biased it toward producing higher quality outputs.

**Question for the audience:** If you wanted to create a LoRA for generating images in the style of a specific artist, what would be your training approach?"

---

## PART 5: MULTILINGUAL CAPABILITIES

### Cell 15-20: Multilingual Prompts (6 minutes)

**[Execute Cells 16, 18, 20 sequentially]**

**Script:**
"This is where things get really interesting. We're going to test the model's multilingual understanding.

**Cell 16 - English Baseline:**
[Execute and show image]

'A handsome Asian man wearing a dark gray suit...' This is our baseline.

**Cell 18 - Korean:**
[Execute and show image]

Same prompt, but in Korean. Notice something fascinating here.

**For Beginners:**
The model understands Korean even though it's primarily trained on images and English text. How? The text encoder uses multilingual representations.

**For Intermediate Students:**
The text encoder is likely based on a multilingual transformer (similar to XLM-RoBERTa or multilingual BERT):
- Trained on text from 100+ languages
- Learns shared embedding space
- Concepts like 'suit' or 'flowers' have similar representations across languages

**Cell 20 - Kannada:**
[Execute and show image]

Now Kannada—a Dravidian language spoken in Karnataka. Still works!

**For Advanced Students:**
This demonstrates **zero-shot cross-lingual transfer**:
1. Text encoder trained on multilingual corpus
2. Image-text alignment learned primarily in English
3. Shared semantic space enables transfer

The architecture likely uses:
```
Text (any language) → Multilingual Encoder → Shared Embedding Space → Image Generation
```

Key concepts:
- **Tokenization:** Likely using BPE or WordPiece across languages
- **Embedding alignment:** Languages clustered by semantic similarity
- **Attention mechanisms:** Language-agnostic at deeper layers

**Limitations to discuss:**
- Performance likely better in high-resource languages
- Cultural concepts may be biased toward Western/Chinese imagery
- Complex language-specific idioms might not translate well

**Real-world application:** This is crucial for companies operating globally. One model serves all markets!

**Fun experiment:** Try prompts mixing languages or using language-specific cultural references."

---

### Cell 21-24: Model Cleanup (2 minutes)

**[Execute Cells 23-24]**

**Script:**
"Before moving to the next model, let's see how many parameters we loaded and clean up memory.

[Execute Cell 23]

See that number? That's approximately **86 billion parameters** loaded in memory.

To put this in perspective:
- Storage: ~170 GB in bfloat16
- Memory during inference: Even more with activations
- On a 24GB consumer GPU: Impossible without extreme quantization

[Execute Cell 24]

```python
del qwen_image
torch.cuda.empty_cache()
```

**For Intermediate Students:**
We're explicitly:
1. Deleting the Python object
2. Calling CUDA cache cleanup
3. Freeing GPU memory for the next model

**For Advanced Students:**
PyTorch's memory management:
- Lazy deallocation: memory isn't freed immediately
- `empty_cache()` releases cached memory back to GPU
- Important: This doesn't free memory if tensors are still referenced

In production, you'd use techniques like:
- Model parallelism (split across GPUs)
- Pipeline parallelism (different layers on different GPUs)
- CPU offloading for activation checkpointing

**Key point:** The MI300X's 192GB lets us be wasteful. In production on smaller GPUs, memory management becomes critical."

---

## PART 6: IMAGE EDITING

### Cell 25-27: Loading Edit Model (4 minutes)

**[Execute Cell 27]**

**Script:**
"Now we're loading a different model: **Qwen-Image-Edit**. This is specialized for editing and inpainting.

**For Beginners:**
While the base model generates images from scratch, this one can:
- Modify existing images
- Extend images (outpainting)
- Fill in missing parts (inpainting)
- Maintain consistency with the original

**For Intermediate Students:**
Notice we're reusing some components:

```python
ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
```

The text encoder and VAE are the same—only the transformer is different.

**Why?**
- Text encoding doesn't change between tasks
- VAE encodes/decodes images the same way
- Only the diffusion process (transformer) needs task-specific training

This is **modular architecture**—reuse what you can!

**For Advanced Students:**
The Edit model likely uses:
- **Conditional diffusion:** Image input as additional conditioning
- **ControlNet or similar:** Preserves structure from input image
- **Attention masking:** Focuses on editable regions

Architecture comparison:
```
Base Model:  Text → Transformer → Image
Edit Model:  Text + Input Image → Transformer + Control → Edited Image
```

The processor_config suggests additional image preprocessing for edit-specific features like masks or control signals.

**Memory consideration:** We just freed the base model and loaded the edit model. On consumer hardware, you'd have to choose one or the other. Here? We could load both simultaneously!"

---

### Cell 28-29: Basic Outpainting (4 minutes)

**[Execute Cell 29]**

**Script:**
"Let's do outpainting—extending the portrait we created earlier into a full-body shot with a forest background.

**For Beginners:**
Look at this prompt carefully:

```python
negative_prompt = 'Make the character's fingers mutilated...'
```

This is a **negative prompt**—things we DON'T want. The model learns to avoid these features.

**For Intermediate Students:**
Negative prompts use classifier-free guidance:

```
output = base_output + guidance_scale × (text_cond_output - negative_cond_output)
```

We're essentially pushing the output away from negative prompt features.

**For Advanced Students:**
Classifier-free guidance (CFG):
1. Train model with and without text conditioning (dropout text with probability p)
2. At inference:
   ```
   ε_pred = ε_uncond + s(ε_text - ε_negative)
   ```
   where s is guidance scale

The negative prompt increases:
- Diversity in generation
- Control over unwanted features
- Image quality by avoiding common artifacts

[Show generated image]

**Point out:** The face changed! The outpainting worked, but we lost facial consistency. This is a common problem in image editing.

**Discussion:** Why did this happen? The model doesn't have strong facial identity preservation without additional training."

---

### Cell 30-31: Face-to-Person LoRA (5 minutes)

**[Execute Cell 31]**

**Script:**
"To fix the consistency problem, we're loading another LoRA: **Qwen-Image-Edit-F2P** (Face-to-Person).

This LoRA was specifically trained to maintain facial identity when extending portraits to full-body images.

[Execute and show result]

**Compare the images:**
- Before F2P LoRA: Different face
- After F2P LoRA: Same facial features preserved!

**For Intermediate Students:**
This LoRA learned:
1. Facial feature extraction
2. Identity preservation across different scales
3. Consistent rendering when adding body/background

Think of it as adding 'facial memory' to the model.

**For Advanced Students:**
This likely uses techniques similar to:
- **IP-Adapter:** Injects image prompt features via attention
- **FaceID:** Extracts facial embeddings (like ArcFace)
- **Cross-attention injection:** Facial features guide body generation

Architecture:
```
Input Face → Feature Extractor → Embedding
Embedding + Text → Modified Attention Layers → Consistent Output
```

The LoRA modifies specific attention layers to:
```
Q = W_q × X + B_lora × A_lora × X
K = W_k × [X; Face_Embedding]
```

**Real-world use case:** Virtual try-on, personalized avatars, character consistency in game asset generation.

**Technical challenge question:** How would you extend this to preserve identity across multiple people in a scene?"

---

### Cell 32-34: Cleanup (1 minute)

**[Execute Cells 33-34 quickly]**

**Script:**
"Same pattern—count parameters, clean up memory. We're getting ready for the final model."

---

## PART 7: MULTI-IMAGE EDITING

### Cell 35-36: Loading Multi-Image Edit Model (4 minutes)

**[Execute Cell 36]**

**Script:**
"Final model: **Qwen-Image-Edit-2509**. This one can merge multiple images into a single coherent scene.

**For Beginners:**
We're going to combine:
- The woman in the forest
- The man with flowers
Into a single romantic scene!

**For Intermediate Students:**
This is compositional editing—much harder than single-image editing because we need:
- Identity preservation for both subjects
- Spatial reasoning (where should they be?)
- Lighting consistency
- Natural pose generation
- Background blending

**For Advanced Students:**
Multi-image editing requires:
1. **Multiple conditioning paths:**
   ```
   Transformer([Image1_Features, Image2_Features, Text_Features])
   ```

2. **Spatial layout understanding:**
   - Layout encoding (bounding boxes or masks)
   - Depth-aware composition
   - Occlusion handling

3. **Feature disentanglement:**
   - Separate identity from pose
   - Separate lighting from geometry
   - Combine semantically

Possible architectures:
- Multiple ControlNet branches
- Multi-scale feature injection
- Reference-based synthesis
- Layout-conditioned generation

**State-of-the-art techniques:**
- GLIGEN: Grounded language-to-image generation
- Composable Diffusion: Compositional visual generation
- LayoutDiffusion: Controllable layout synthesis

**Challenge:** Maintaining global coherence while respecting local constraints from multiple sources."

---

### Cell 37-38: Merging Two People (5 minutes)

**[Execute Cell 38]**

**Script:**
"Now for the grand finale! The prompt is in Korean:

'이 사랑 넘치는 부부의 포옹하는 모습을 찍은 사진을 생성해 줘'

Which means: 'Generate a photo of this loving couple embracing.'

Notice:
- Korean prompt ✓
- Two input images ✓
- Complex interaction (embracing) ✓

[Execute and wait]

[Show result]

**For All Students:**
This is remarkable! The model:
1. Understood the Korean prompt
2. Extracted identities from both images
3. Reasoned about spatial positioning for an embrace
4. Generated natural poses and lighting
5. Created a coherent background

**For Advanced Students:**
This demonstrates:
- **Compositional generalization:** Combining learned concepts
- **Spatial reasoning:** Understanding pose relationships
- **Multi-modal fusion:** Text + Image1 + Image2 → New Image

The latent space must encode:
- Identity vectors (from each face)
- Pose semantics ('embracing')
- Scene composition rules
- Lighting and style consistency

**Failure modes to consider:**
- Identity mixing (blended faces)
- Impossible poses
- Lighting inconsistencies
- Background artifacts

**Production deployment considerations:**
- Inference time per image
- Batch processing capabilities
- Quality consistency across prompts
- Handling edge cases (many people, complex scenes)

**Question for discussion:** What safeguards would you implement if deploying this as a product?"

---

### Cell 39-41: Final Parameter Count (3 minutes)

**[Execute Cells 39, 41]**

**Script:**
"Let's count parameters across all three models we've used.

[Show total]

**~86 billion parameters** across three models!

**For Beginners:**
To train these from scratch would require:
- Millions of images
- Thousands of GPU-hours
- Hundreds of thousands of dollars
- Months of time

**For Intermediate Students:**
Training costs estimation:
- Data: Web-scale image-text pairs (LAION-5B scale)
- Compute: ~10,000 A100 GPU hours ≈ $200,000
- Storage: Petabytes of training data
- Engineering: Distributed training infrastructure

**For Advanced Students:**
Training challenges:
1. **Data:** LAION-5B, COYO-700M, proprietary datasets
2. **Compute:** Multi-node distributed training
3. **Optimization:** AdamW with gradient clipping, learning rate scheduling
4. **Stability:** Mixed precision training, gradient checkpointing
5. **Evaluation:** FID, CLIP score, human evaluation

**Why AMD MI300X matters for this:**
- 192GB enables larger batch sizes
- Larger batches → better gradient estimates → faster convergence
- Can train larger models without model parallelism
- Memory for activation checkpointing without recomputation

**Research question:** What's the next frontier? Some ideas:
- Video generation (temporal consistency)
- 3D generation
- Multi-modal (text + audio + video)
- Real-time generation"

---

## PART 8: TRAINING CUSTOM LoRA

### Cell 44-46: Dataset Preparation (4 minutes)

**[Execute Cell 46]**

**Script:**
"Now we switch from **inference** to **training**. We're going to teach the model about a specific dog!

[Show dataset images]

**For Beginners:**
We have 5 images of the same dog with descriptions:
- 'a dog on the road'
- 'a dog on the grass'
- etc.

We'll train the model to understand 'a dog' means THIS specific dog.

**For Intermediate Students:**
This is **concept learning** or **subject-driven generation**:
- Small dataset (5 images)
- Identity preservation
- Generalization to new contexts

The CSV contains prompt-image pairs. We'll train the model to:
```
Minimize: Loss(Generated_Image('a dog'), Real_Image(dog))
```

**For Advanced Students:**
This is similar to DreamBooth or Textual Inversion:

**DreamBooth approach:**
- Fine-tune the full model (or LoRA) on subject images
- Use rare token identifier (e.g., 'sks dog')
- Class-specific prior preservation

**Textual Inversion approach:**
- Learn new token embedding
- Freeze model, only train embedding vector

We're using LoRA, so:
- Freeze base model (86B parameters)
- Train low-rank adapters (~millions of parameters)
- Much faster, requires less data

**Dataset considerations:**
- `dataset_repeat=50`: Each image seen 50 times
- Diversity important (different poses, lighting, backgrounds)
- Risk of overfitting vs. memorization trade-off

**Training objective:**
```
L = L_diffusion + λ × L_prior_preservation
```

Where:
- L_diffusion: Denoising loss on subject images
- L_prior_preservation: Maintain general knowledge"

---

### Cell 47-53: Training Configuration (6 minutes)

**[Show Cell 53 but don't execute yet]**

**Script:**
"Let's break down this training command. It's dense, but every parameter matters.

**For Beginners:**
We're configuring:
- What data to use
- How to train
- Where to save results

**For Intermediate Students:**
Key parameters:

```bash
--dataset_base_path dataset
--dataset_metadata_path dataset/metadata.csv
```
Data location and annotations.

```bash
--max_pixels 1048576
```
1 megapixel max (1024×1024). Higher = better quality but slower.

```bash
--dataset_repeat 50
```
Augmentation: Each image seen 50 times. With 5 images = 250 training samples.

```bash
--learning_rate 1e-4
```
Adam optimizer step size. Too high = unstable, too low = slow.

```bash
--num_epochs 1
```
One pass through all data. With repeat=50, that's 50 passes per image.

**For Advanced Students:**
Let's go deeper:

```bash
--lora_base_model "dit"
```
Apply LoRA only to the diffusion transformer (DIT), not text encoder or VAE. Why?
- Text encoder: Already generalizes well
- VAE: Task-agnostic (encoding/decoding doesn't change)
- DIT: Where visual concepts are learned

```bash
--lora_target_modules "to_q,to_k,to_v,add_q_proj,..."
```
Specific attention layers to adapt. This targets:
- Query, Key, Value projections (attention mechanism)
- MLP layers (feed-forward networks)
- Modulation layers (adaptive normalization)

**Why these modules?**
- Attention: Controls what the model focuses on
- MLPs: Feature transformations
- Modulation: Conditioning mechanisms

```bash
--lora_rank 32
```
Critical hyperparameter!

Rank 32 means:
- For a 4096×4096 weight matrix:
  - Original: 16M parameters
  - LoRA: 262K parameters (4096×32 + 32×4096)
  - Compression: 98.4%

Higher rank = more expressivity but more parameters.

```bash
--dataset_num_workers 2
```
Parallel data loading. Limited by CPU cores and I/O.

```bash
--find_unused_parameters
```
For distributed training: detect unused parameters in backward pass.

**Training optimization strategy:**
1. Gradient accumulation (if needed for batch size)
2. Mixed precision (bfloat16)
3. Gradient clipping (prevents exploding gradients)
4. Cosine learning rate schedule

**Expected training time:**
- On MI300X: ~10-15 minutes
- On consumer GPU: 30-60 minutes

**Monitoring during training:**
- Loss curve (should decrease)
- Sample generations (visual quality)
- Overfitting signs (loss plateaus but validation worsens)

[Execute Cell 53]

**While training runs:**
'Let's talk about what's happening inside the GPU right now:

1. **Forward pass:** Generate images with current LoRA weights
2. **Loss calculation:** Compare to target images
3. **Backward pass:** Compute gradients
4. **Optimizer step:** Update LoRA weights

This happens for each batch, and we're watching the model learn in real-time!'

[Monitor training output, point out loss values]"

---

## PART 9: TESTING CUSTOM LoRA

### Cell 54-57: Loading Custom LoRA (5 minutes)

**[After training completes, execute Cell 55 and 57]**

**Script:**
"Training complete! Now let's test our custom LoRA.

[Execute Cell 55 - Reload base model]

We're loading the base model again, but this time we'll attach our newly trained LoRA.

[Execute Cell 57]

**For All Students:**
Prompt: 'a dog'

[Show generated image]

**Does it look like our training dog?** Compare with the training images!

**For Intermediate Students:**
What we've achieved:
- Taught the model a new visual concept
- Using only 5 images
- In ~15 minutes
- Without modifying the base model

This is the power of LoRA for rapid iteration and customization.

**For Advanced Students:**
**Evaluation metrics to consider:**

1. **Identity preservation:**
   - Visual similarity to training images
   - Perceptual loss (LPIPS)
   - Face recognition similarity (if applicable)

2. **Generalization:**
   - Works in new contexts?
   - Different poses, lighting, backgrounds?

3. **Quality:**
   - FID (Fréchet Inception Distance)
   - IS (Inception Score)
   - Human preference studies

**Failure modes:**
- **Underfitting:** Doesn't capture identity (rank too low, learning rate too low)
- **Overfitting:** Memorizes training images exactly (rank too high, too many epochs)
- **Mode collapse:** All outputs look identical
- **Catastrophic forgetting:** Model loses general capabilities

**Techniques to improve:**
- More diverse training data
- Regularization (weight decay, dropout)
- Prior preservation (train on both specific and general images)
- Augmentation (crops, flips, color jitter)
- Hyperparameter tuning (rank, learning rate, epochs)"

---

### Cell 58-59: Generalization Test (4 minutes)

**[Execute Cell 59]**

**Script:**
"Final test: Can the model generalize to new contexts?

Prompt: 'a dog is jumping'

We never showed the model this dog jumping—let's see if it understands!

[Show result]

**Analysis:**
- Is it the same dog?
- Is the pose correct (jumping)?
- Background appropriate?
- Overall quality?

**For Advanced Students:**
This tests **compositional generalization:**

```
P(dog_identity ∩ jumping_pose) ≠ P(dog_identity) × P(jumping_pose)
```

The model must:
1. Retrieve learned dog identity from LoRA weights
2. Understand 'jumping' action from base model knowledge
3. Compose them spatially and physically plausibly

**Factors affecting generalization:**
- **Training data diversity:** More varied poses → better generalization
- **LoRA rank:** Higher rank can capture more nuanced features
- **Base model quality:** Stronger base = better composition
- **Prompt engineering:** Specific vs. abstract descriptions

**Research frontier:**
- Few-shot learning (even fewer examples)
- Zero-shot customization (describe appearance in text)
- Multi-concept composition (multiple LoRAs simultaneously)
- Cross-domain adaptation (sketch to photo, cartoon to realistic)

**Industry application:**
- Product photography (showcase products in different settings)
- Character consistency in games/movies
- Personalized content generation
- Virtual try-on applications"

---

## PART 10: CONCLUSION & WRAP-UP

### Cell 60-61: Conclusion (5 minutes)

**[Show cells, don't execute]**

**Script:**
"Let's recap what we accomplished today:

### What We Did:
1. ✅ Loaded and ran an **86 billion parameter** model
2. ✅ Generated images from text in **multiple languages**
3. ✅ Enhanced quality using **LoRA adapters**
4. ✅ Edited and extended images while preserving identity
5. ✅ Merged multiple subjects into a single scene
6. ✅ Trained a **custom LoRA** from just 5 images
7. ✅ Tested generalization to new contexts

### All on a **single AMD Instinct MI300X GPU**!

---

### Key Takeaways by Level:

**For Beginners:**
- AI models can understand text in multiple languages
- Specialized adaptations (LoRA) make customization easy
- Powerful hardware enables previously impossible tasks

**For Intermediate Developers:**
- Modular architecture (transformer, text encoder, VAE)
- Low-rank adaptation is memory-efficient and fast
- Negative prompts control output quality
- Environment configuration matters (cache, offline mode)

**For Advanced Researchers:**
- Latent diffusion architecture and its components
- Classifier-free guidance and conditioning mechanisms
- LoRA mathematics and architectural considerations
- Training optimization strategies
- Compositional generalization challenges

---

### Why AMD MI300X Matters:

**192 GB HBM3 Memory:**
- Load multiple massive models simultaneously
- Larger batch sizes = faster training
- No need for aggressive model parallelism
- Room for larger context windows and higher resolution

**ROCm Ecosystem:**
- Open-source (contribute and customize)
- PyTorch native support
- Growing library of optimized kernels
- Competitive with CUDA ecosystem

**Performance:**
- High memory bandwidth (5.2 TB/s)
- Optimized for AI workloads
- Efficient power consumption
- Enterprise reliability

---

### Next Steps for You:

**Beginner Projects:**
1. Generate images with different prompts and styles
2. Experiment with negative prompts
3. Try different LoRAs from HuggingFace
4. Generate images in your native language

**Intermediate Projects:**
1. Train custom LoRAs for your own subjects
2. Implement image editing pipelines
3. Combine multiple LoRAs
4. Build a simple web interface (Gradio, Streamlit)

**Advanced Projects:**
1. Implement custom conditioning mechanisms
2. Research alternative LoRA architectures
3. Optimize inference speed (quantization, distillation)
4. Develop multi-stage editing pipelines
5. Contribute to DiffSynth-Studio or ROCm

**Research Directions:**
- Video generation with temporal consistency
- 3D-aware image generation
- Controllable generation (pose, lighting, style)
- Efficient training techniques
- Safety and ethics (deepfakes, copyright)

---

### Resources:

**Documentation:**
- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- AMD ROCm: https://rocmdocs.amd.com/
- Qwen-Image: https://huggingface.co/Qwen

**Learning:**
- LoRA paper: https://arxiv.org/abs/2106.09685
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- DreamBooth: https://arxiv.org/abs/2208.12242

**Community:**
- AMD AI Developer Program: https://www.amd.com/en/developer/ai-dev-program.html
- HuggingFace Forums
- ROCm GitHub Issues

---

### Questions to Ponder:

1. **Ethics:** What guardrails should exist for personalized image generation?
2. **Bias:** How do we ensure models represent all demographics fairly?
3. **Copyright:** Who owns AI-generated images trained on copyrighted data?
4. **Accessibility:** How can we make these tools available to everyone?
5. **Environment:** What's the carbon footprint of training/inference?

---

### Final Thoughts:

**We're at an inflection point in AI:**
- Models are becoming multimodal (text, image, video, audio)
- Customization is becoming democratized
- Hardware is catching up to our ambitions
- Open-source is driving innovation

**Your generation will:**
- Build the next generation of creative tools
- Solve problems we haven't imagined yet
- Democratize access to powerful AI
- Navigate the ethical implications

**The tools you learned today** are just the beginning. The real power is in how you apply them.

---

### Q&A Session

**Opening for questions:**
'I'll take questions now. Here are some topics we can dive deeper into:

**Technical:**
- Architecture details
- Training optimization
- Production deployment
- Performance tuning

**Practical:**
- Setting up your own environment
- Finding datasets
- Hyperparameter tuning
- Debugging common issues

**Conceptual:**
- How diffusion models work mathematically
- LoRA vs other fine-tuning methods
- Comparison to other image generation approaches
- Future of multimodal AI

**Career:**
- ML engineering roles
- Research opportunities
- Relevant coursework
- Building a portfolio

Who has the first question?'

---

### Closing (1 minute):

**Script:**
'Thank you all for your attention and engagement today!

**Remember:**
- You have access to these notebooks
- The AMD AI Developer Program offers free cloud credits
- This technology is evolving rapidly—stay curious!
- The best way to learn is to build

**Call to action:**
1. Run this notebook yourself
2. Train a custom LoRA on something meaningful to you
3. Share your results with the community
4. Keep pushing the boundaries

**Final quote:**
'The best way to predict the future is to invent it.' - Alan Kay

You have the tools, the knowledge, and the hardware. Now go build something amazing!

Thank you, and feel free to reach out with questions anytime!'

---

## APPENDIX: BACKUP SLIDES & TALKING POINTS

### If Students Ask About Alternatives:

**Stable Diffusion Comparison:**
- Qwen-Image: 86B parameters, better quality, multilingual
- Stable Diffusion: 1-2B parameters, faster, more community models
- Trade-offs: Quality vs. Speed, Capabilities vs. Accessibility

**DALL-E / Midjourney:**
- Closed-source, API-only
- Great quality but less control
- Can't fine-tune or customize easily
- This workshop shows open-source alternatives

**FLUX / Imagen:**
- FLUX: Open-source, similar architecture, strong performance
- Imagen: Google's model, text encoder uses T5
- Different design choices, similar capabilities

### If Demo Fails:

**Backup talking points:**
- Explain what SHOULD happen
- Show pre-generated results
- Discuss troubleshooting approach:
  - Check GPU memory
  - Verify model loading
  - Debug error messages
  - Check environment variables
- Use failure as teaching moment about production reliability

### If Running Out of Time:

**Priority order:**
1. Must show: Cell 8, 10, 14, 27, 31, 38 (core capabilities)
2. Should show: Cell 55, 57, 59 (custom training)
3. Nice to have: Detailed explanations, Q&A

**Speed-up strategies:**
- Skip some markdown cells
- Run cells during earlier explanations
- Pre-run time-consuming cells
- Focus on results over process

### Advanced Topics If Time Permits:

**Quantization:**
- INT8, INT4 quantization for deployment
- bitsandbytes library
- QLoRA (quantized LoRA training)

**Distributed Training:**
- Model parallelism
- Pipeline parallelism
- Data parallelism
- FSDP (Fully Sharded Data Parallel)

**Production Deployment:**
- Serving frameworks (TorchServe, TensorRT)
- Batching strategies
- Caching mechanisms
- Load balancing

---

**END OF PRESENTER SCRIPT**

*Remember: The goal is to inspire students while teaching fundamentals. Balance theory with practice, and always connect back to real-world applications!*
