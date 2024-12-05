# 🎬🎨 VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement 

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://video-repair.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2411.15115-b31b1b.svg)](https://arxiv.org/pdf/2411.15115.pdf)   

#### [Daeun Lee](https://daeunni.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Jaemin Cho](https://j-min.io), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)    


<br>
<img width="950" src="image/teaser_final_verylarge_v2.gif"/>
<br>

✨ **VideoRepair** can *(1) **detect** misalignments by generating fine-grained evaluation questions and answering, (2) **plan** refinement, (3) **decompose** the region* and finally *(4) conduct **localized refinement**.*      

## 🔧 Setup

### Environment Setup 
You can install all packages from ```requirements.txt```. 
```shell
conda create -n videorepair python==3.10
conda activate videorepair
pip install -r requirements.txt 
```
Additionally, for Semantic-SAM, you should install detectron2 like below: 
```shell 
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
```

### OpenAI API Setup 
Our VideoRepair is based on GPT4 / GPT4o, so you need to setup your Azure OpenAI API config in the below files. 
You can set your own API infomation in `config.ini`. 

```python
[openai]
azure_endpoint = your endpoint   
api_key = your key 
api_version = your version 
```

### Download Models 
You can download pre-trained models here. 
- [T2V-turbo](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt)
- [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
- [MolmoE-1B-0924](https://huggingface.co/allenai/MolmoE-1B-0924)
- [Semantic-SAM (L)](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth)
- BLIP-BLUE 
```shell
git lfs install
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
```

Next, please locate all downloaded models in the  `./checkpoints` directory! The code structure will like below: 
```bash
./checkpoints
    ├── blip2-opt-2.7b
    ├── t2v-turbo 
    │   ├── unet_lora.pt
    │   ├── inference_t2v_512_v2.0.yaml     # downloaded from T2V-turbo official repo 
    ├── VideoCrafter
    │   ├── model.ckpt

./SemanticSAM/checkpoint
    ├── ssam
    │   ├── swinl_only_sam_many2many.pth
```

## 🎨 Apply to your own prompt  
We provide demo (`run_demo.sh`) for your own prompt! This demo use `main_iter_demo.py`. 
```shell
output_root="your output root"
prompt="your own prompt"

CUDA_VISIBLE_DEVICES=1,2 python main_iter_demo.py --prompt="$prompt" \
                                                    --model="t2vturbo" \              # base t2v-model 
                                                    --output_root="$output_root" \
                                                    --seed=123 \                      # global random seed (use for initial video generation) 
                                                    --load_molmo \            
                                                    --selection_score='dsg_blip' \    # video ranking method 
                                                    --round=1 \
                                                    --seed=369                        # localized generation seeds 
```

## 🌿 Apply to Benchmark 
VideoRepair is tested on [EvalCrafter](https://github.com/EvalCrafter/EvalCrafter) and [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench). 

We provide our $dsg^{obj}$ questions in `./datasets`. The structure is like below: 
```bash
./datasets
    ├── compbench
    │   ├── consistent_attr.json
    │   ├── numeracy.json
    │   ├── spatial_relationship.json
    ├── evalcrafter
    │   ├── dsg_action.json
    │   ├── dsg_color.json
    │   ├── dsg_count.json
    │   ├── dsg_none.json
```
Based on above question set, you can run benchmarks as follows: 
```bash
output_root="your output path"                # output path 
eval_sections=("count", "action", "color")                       # eval dimension for each benchmark (e.g., count, )

for section in "${eval_sections[@]}"
do
    CUDA_VISIBLE_DEVICES=1,2,3 python main_iter.py \
                        --output_root="$output_root" \
                        --eval_section="$section" \
                        --model='t2vturbo' \              # t2v model backbone 
                        --selection_score='dsg_blip' \    # video ranking metric 
                        --seed=123 \                      # random seed 
                        --round=1 \                       # iteration round 
                        --k=10 \                          # number of video candidates 
done
```


## 📝 TODO List
- [ ] Release the demo code.
- [x] Release the benchmark generation code.


## 📚 BibTeX

💗 If you enjoy our VideoRepair and find some beneficial things, citing our paper would be the best support for us! 

```
@article{lee2024videorepair,
  title={VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement},
  author={Lee, Daeun and Yoon, Jaehong and Cho, Jaemin and Bansal, Mohit},
  journal={arXiv preprint arXiv:2411.15115},
  year={2024}
}
```


