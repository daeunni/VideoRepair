# 🎬🎨 VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement 

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://video-repair.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2402.08712-b31b1b.svg)](https://arxiv.org/pdf/2411.xxxx.pdf)   

<img width="928" alt="image" src="">  

## Abstract
Recent text-to-video (T2V) diffusion models have demonstrated
impressive generation capabilities across various
domains. However, these models often generate videos that
have misalignments with text prompts, especially when the
prompts describe complex scenes with multiple objects and
attributes. To address this, we introduce VIDEOREPAIR, a
novel model-agnostic, training-free video refinement framework
that automatically identifies fine-grained text-video
misalignments and generates explicit spatial and textual
feedback, enabling a T2V diffusion model to perform targeted,
localized refinements. VIDEOREPAIR consists of four
stages: In (1) video evaluation, we detect misalignments by
generating fine-grained evaluation questions and answering
those questions with MLLM. In (2) refinement planning, we
identify accurately generated objects and then create localized
prompts to refine other areas in the video. Next, in (3)
region decomposition, we segment the correctly generated
area using a combined grounding module. We regenerate
the video by adjusting the misaligned regions while preserving
the correct regions in (4) localized refinement. On
two popular video generation benchmarks (EvalCrafter and
T2V-CompBench), VIDEOREPAIR substantially outperforms
recent baselines across various text-video alignment metrics.
We provide a comprehensive analysis of VIDEOREPAIR components
and qualitative examples
