# Research Log

How a C++ inference engine for a video VLM went from 18.85 tok/s (Python HuggingFace) to 31.3 tok/s (1.66x) on an M3 Max, and what I learned about optimising a hybrid SSM/attention model that the Qwen3-4B project did not teach me. Including the part where my benchmark was wrong and the number I reported first was too high.

## The starting point

I had just finished [qwen3-mlx](../inference/), a C++ engine for Qwen3-4B. Pure transformer, pure attention, 125.77 tok/s. The interesting finding there was that five out of six optimisation hypotheses were wrong, and the one that worked was a weight layout fix you could not see in a kernel trace. I wanted to know if the same methodology: measure the data before the code. That would hold on a different architecture.

NemoStation/Marlin-2B is a video VLM fine-tuned from Qwen3.5-2B. It takes video frames and produces structured captions with timestamps. The architecture is not a pure transformer: 18 of 24 layers are GatedDeltaNet (a recurrent SSM), only 6 are attention. It has a ViT vision encoder, a 3D patch embedding, and a spatial merger. The Hugging Face card describes it as "a fine-tune of Qwen3.5-2B with the video-capable visual tower kept intact." That turned out to be about 30% of what I needed to know.

## The Python baseline

Before writing any C++, I profiled the Python HuggingFace path on the same hardware.

M3 Max, 40 GPU cores, 48 GB unified memory, 400 GB/s bandwidth. Marlin-2B bf16, 4-video synthetic corpus (10-30s, 448x448), greedy decoding:

| Video | Prompt tokens | Decode tok/s |
|---|---|---|
| scene_changes (10s) | 2,086 | 23.3 |
| motion (15s) | 3,111 | 18.3 |
| events (20s) | 4,136 | 19.1 |
| color_sweep (30s) | 6,186 | 14.7 |
| **Mean** | | **18.85** |

Three runs, identical SHA-256 hashes across all outputs. Determinism verified.

The Python path was running the torch fallback. No flash-linear-attention, no causal-conv1d. The transformers warning said: "The fast path is not available because one of the required library is not installed." Those libraries are CUDA-only and do not run on MPS.

## The investigation

I used a 5-level bandwidth-first investigation, same methodology as the Qwen3-4B project. Each level produces a verdict before advancing.

### Level 1: Where do the bytes go?

I read the safetensors headers and computed per-token decode bandwidth. The model is 5.44 GB in bf16. During decode, the vision encoder is not read (prefill only). The LLM weights read per token:

| Component | MB/token | % |
|---|---|---|
| MLP (24 layers) | 1,812 | 48% |
| lm_head (tied) | 1,017 | 27% |
| SSM (18 layers) | 758 | 20% |
| Attention (6 layers) | 176 | 5% |
| **Total** | **3,764** | |

Theoretical bf16 ceiling: `400 GB/s / 3.76 GB = 106 tok/s`. Measured: 18.85 tok/s. Efficiency: 17.7%.

But 17.7% included prefill overhead. Isolating pure decode by measuring the delta between 32-token and 256-token runs: pure decode was actually 34 tok/s, achieving 128 GB/s (32% of bandwidth). Still low, but twice what the end-to-end number suggested.

**Verdict: FOUND.** bf16 everywhere. MLP + lm_head = 75% of decode bandwidth.

### Level 2: Why 32% efficiency?

The GatedDeltaNet torch fallback path is a Python for-loop over sequence length with 6 unfused tensor ops per step per layer. `causal_conv1d_fn: None`. Everything cast to fp32 inside the recurrence. Estimated ~433 kernel dispatches per decode token.

I read the actual source code of `torch_recurrent_gated_delta_rule` from the transformers library. It is a literal `for i in range(sequence_length)` loop.

**Verdict: FOUND.** Dispatch overhead from the unoptimised fallback path.

### Level 3: Is this fixable?

Three research agents verified feasibility:

1. **mlx-lm already supports Qwen3.5 GatedDeltaNet** with native Metal kernels. Not a novelty risk.
2. **ZMLX has a fused DeltaNet decode kernel** for MLX. +7.5% on M4 Max. Proves Metal kernel fusion works for this recurrence.
3. **Q4 quantisation on SSM layers needs mixed precision.** MambaQuant (arXiv 2501.13484) found heavy-tailed outliers in SSM gate/output projections. Uniform Q4 degrades quality. Q4_K_M (which keeps sensitive layers at 6-bit) already approximates this.

## What worked

Two things. Out of three attempts.

### 1. Quantise the lm_head (same as Qwen3-4B)

The lm_head is `[248320, 2048]` bf16 = 1,017 MB read every token. Quantising to Q4_g64 reduces it to ~270 MB. This is the exact same optimisation that worked on Qwen3-4B, for the exact same reason: the tied embedding table dominates lm_head bandwidth.

```
bf16 baseline:     16.1 tok/s
Q4 lm_head:        20.5 tok/s  (+27%)
```

### 2. Quantise the MLP projections

The 24 MLP layers (gate/up/down projections) are 48% of decode bandwidth. All three projections per layer are large dense matmuls with no SSM sensitivity concerns. Quantising to Q4_g64:

```
Q4 lm_head only:   20.5 tok/s
Q4 MLP + lm_head:  35.7 tok/s  (+74%, 1.89x Python)
```

The SSM weights (in_proj_qkv, in_proj_a/b/z, conv1d, A_log, dt_bias) stay bf16. This is the mixed-precision approach MambaQuant recommends: quantise the MLP and attention projections, leave the SSM-critical weights alone.

## What did not work

### 3. Compiled recurrence (mx::compile)

The decode-path recurrence does ~36 elementwise ops per SSM layer: expand_dims, multiply, sum, subtract, sigmoid, exp, softplus. I wrapped the recurrence + RMSNormGated in `mx::compile(fn, true)` (shapeless compilation) to fuse them into a single compiled kernel.

Neutral to slightly negative. The elementwise ops are not the bottleneck. The quantised matmuls are. MLX's lazy evaluation already batches the small ops reasonably well without explicit compilation.

ZMLX's custom Metal kernel for the same recurrence gets +7.5% on an already-optimised mlx-lm baseline. Same conclusion: the matmuls dominate, the recurrence is not where the time goes.

## The two correctness bugs

Building the engine was straightforward. Getting correct output was not. Two bugs cost most of the debugging time.

### Bug 1: RMS norm +1 offset

Symptom: the engine ran, generated tokens, but produced garbage (repeating newlines). Layer-by-layer comparison showed embeddings matched exactly but layer 0 output diverged completely.

I dumped the norm output at layer 0. Python std: 1.12. C++ std: 0.18. The ratio is roughly `1 / (1 + mean_weight)`.

Cause: Qwen3.5's `Qwen3_5RMSNorm` uses `output = norm(x) * (1.0 + self.weight)`. The weight parameter is initialised to zeros and trained to small values (mean ~0.096). Without the +1, you multiply by 0.096 instead of 1.096.

This applies to input norms, post-attention norms, final norm, and Q/K norms. It does *not* apply to `Qwen3_5RMSNormGated` inside the SSM layers, which initialises weight to ones and uses plain `weight * norm(x)`.

Fix: one line. `return x * rsqrt(var + eps) * (1.0f + weight)`.

### Bug 2: Attention q/gate split

Symptom: after fixing the norm, layers 0-2 (SSM) matched Python perfectly but layer 3 (first attention layer) diverged.

I dumped the q and gate projections. Python q_mean: -0.033. C++ q_mean: -1.449. The gate stats were equally wrong.

Cause: the `q_proj` weight is `[4096, 2048]`, producing both query and output gate. The correct split is: reshape to `[B, S, num_heads, 2 * head_dim]`, then chunk on the last dimension. I was splitting the flat `[B, S, 4096]` in half, which interleaves query and gate values across heads.

Fix: add a reshape before the split.

After both fixes, all 24 layers matched Python within bf16 rounding tolerance. Top-1 token: `<think>` (id 248068, logit 33.39 C++ vs 33.50 Python).

## The benchmark that lied

This is the part I did not expect.

After the two LLM correctness bugs were fixed, I ran the benchmark. 35.7 tok/s on the synthetic corpus. 1.89x Python. I wrote it up, committed it, reported it as the final number.

Then I tried CaReBench: 1,000 real videos with ground-truth captions. The engine produced "grids of tan-colored cubes" and "surreal visual compositions" on videos of women doing eyebrow tutorials and athletes throwing javelins. The output was structurally correct (Scene + Events format) but semantically garbage.

The synthetic corpus had hidden three bugs in the vision pipeline:

### Bug 3: Frame decoding mismatch

FFmpeg's `fps=2` video filter and torchcodec's uniform sampling produce different frames from the same video. Not different timestamps, different actual frames. Pixel values at the same coordinates differed by 0.6 on a [-1, 1] scale. The model is sensitive to the exact frame content.

I verified this by saving Python's exact frames and comparing pixel-by-pixel. The fix was switching from the `fps` filter to per-frame seeking at explicit timestamps (0.0s, 0.5s, 1.0s, ...) to match torchcodec's sampling.

The synthetic corpus did not catch this because solid-colour frames look the same regardless of which exact frame you sample.

### Bug 4: Patch ordering mismatch

The HF processor flattens video patches in 2x2 spatial merge groups: `(h=0,w=0), (h=0,w=1), (h=1,w=0), (h=1,w=1), (h=0,w=2), ...`. Our conv3d output was in row-major order: `(h=0,w=0), (h=0,w=1), (h=0,w=2), ...`. The ViT processed patches in the wrong spatial arrangement.

I found this by brute-force searching: for each processor patch, scan all spatial positions in our frames until the pixel values match. Processor patch 100 was at spatial position (h=0, w=50), not our (h=1, w=20).

The synthetic corpus did not catch this because at 448x448 (28x28 patches), the merge-group ordering and row-major ordering are similar enough that uniform content produces similar ViT output either way.

### Bug 5: Missing ViT rotary position encoding

The Qwen3.5 ViT uses 2D rotary position embeddings (row and column frequencies) applied to query and key in every attention block. Without them, the ViT attention is purely content-based with no spatial awareness. It cannot distinguish where a patch is in the frame.

I implemented the rotary encoding: compute (row, col) position IDs in merge-group order, generate inverse frequency vectors, apply `q * cos + rotate_half(q) * sin` in each ViT attention block.

The synthetic corpus did not catch this because solid colours have no spatial structure. The ViT can describe "a red screen" without knowing where any patch is located. It cannot describe "a woman applying eyebrow pencil in front of a mirror" without spatial awareness.

### Why the synthetic benchmark passed

All three bugs share the same failure mode: they are invisible on spatially uniform, low-resolution content.

- Frame decoding: solid colours look the same at any timestamp
- Patch ordering: uniform patches produce similar ViT output regardless of ordering
- Position encoding: no spatial structure means position does not matter

The 448x448 synthetic videos with solid colour blocks were the worst possible test corpus for a vision pipeline. They tested the LLM correctness (which was real; the two norm/gate bugs were genuine) but they did not test vision correctness at all.

The real benchmark is CaReBench at native resolution. The corrected numbers:

## The numbers

| Stage | Synthetic tok/s | CaReBench tok/s | vs Python | What changed |
|---|---|---|---|---|
| Python HuggingFace baseline | 18.85 | ~18.85 | 1.00x | |
| C++ MLX, bf16, unfused | 16.1 | — | 0.85x | Built the engine |
| + Q4 lm_head | 20.5 | — | 1.09x | Level 1 finding |
| + Q4 MLP | 35.7 | — | 1.89x | Level 1 finding |
| + compiled recurrence | 35.5 | — | 1.88x | Dead end |
| + vision pipeline fixes | 31.3 | 19.4 | 1.66x | Bugs 3-5 |

The 35.7 number was real on synthetic video but not meaningful. The 31.3 / 19.4 numbers are real on actual video.

The CaReBench number (19.4 tok/s) is lower than synthetic (31.3) primarily because real video at 1280x720 produces 3,520 spatial patches per temporal group vs 784 at 448x448. That is 4.5x more ViT compute per prefill. The decode speed after prefill is similar.

## What I learned that the Qwen3-4B project did not teach me

**SSM state is different from KV cache.** A KV cache grows linearly with context and you can trim it. SSM recurrent state is fixed-size (512 KB per layer) but cannot be trimmed at arbitrary positions. This breaks prefix caching and speculative decoding, which both assume you can reuse partial state.

**Mixed-precision quantisation is not optional for SSM models.** On a pure transformer you can Q4 everything and check perplexity. On a hybrid SSM/attention model, the SSM gate projections have different numerical properties from the attention projections. MambaQuant documented this; I confirmed it by keeping SSM weights at bf16 while quantising MLP and attention.

**The HF card is necessary but not sufficient.** Every one of the eight gotchas in the README (norm offset, q/gate split, 3D patch embed, depthwise conv1d, vision token structure, SSM state semantics, patch ordering, rotary position encoding) required reading the actual transformers source code or inspecting the safetensors headers. The model card mentions none of them.

**The bandwidth-first methodology transfers.** Different architecture, different model, different modality. Same principle: measure what the GPU reads before changing what the code does. The MLP + lm_head quantisation that gave 1.66x was found at Level 1 of the investigation (safetensors header inspection), not at Level 3 (kernel profiling).

**Synthetic benchmarks can hide vision bugs.** This is the one I will remember. The synthetic corpus validated the LLM pipeline and the quantisation correctly. It did not validate the vision pipeline at all. Three bugs survived because the test content had no spatial structure for the bugs to corrupt. The methodology: measure the data before the code. That was right for the LLM side but I did not apply it to the vision side until CaReBench forced me to. A reference oracle (dump Python's intermediate tensors and compare) would have caught all three in an hour. I spent longer than that chasing the wrong hypotheses.

## What is left

The engine achieves 31.3 tok/s on synthetic and 19.4 tok/s on real video, at roughly 9% of the Q4 theoretical ceiling. The remaining gap is:

1. **SSM projections still bf16** (758 MB/tok, 20% of bandwidth). Quantising these with mixed precision (Q4 for in_proj_qkv/out_proj, Q8 for A_log/dt_bias/conv1d) would reduce decode read by another ~500 MB. Needs perplexity validation on CaReBench.

2. **No weight cache.** Quantisation happens at load time (~2s). Caching to a single safetensors file (like the Qwen3-4B engine) would eliminate this.

3. **Sequential SSM prefill.** The recurrence processes tokens one at a time. FLA's `chunk_gated_delta_rule` parallelises this across chunks but requires a Triton-style kernel.

4. **Missing bilinear position embedding interpolation.** The Python model uses learned bilinear-interpolated position embeddings in addition to rotary. Not yet implemented.

5. **ViT prefill at native resolution is expensive.** CaReBench videos at 1280x720 produce 31,680 patches (vs 7,840 for synthetic 448x448). The ViT processes all of them with full self-attention. Downscaling to match VIDEO_MAX_PIXELS (as qwen-vl-utils does) would trade quality for speed.
