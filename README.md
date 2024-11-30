# 🎬🎨 VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement 

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://video-repair.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2411.15115-b31b1b.svg)](https://arxiv.org/pdf/2411.15115.pdf)   

#### [Daeun Lee](https://daeunni.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Jaemin Cho](https://j-min.io), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)    
🚨 All code will be released by the first week of Dec, stay tuned!    


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
You can find your keys in Azure Portal. We recommend using [python-dotenv](https://github.com/theskumar/python-dotenv) to store and load your keys.

- `DSG/openai_utils.py`
- `DSG/dsg_questions_gen.py`
- `DSG/query_utils.py`
- `DSG/vqa_utils.py`

```python
client = AzureOpenAI(
            azure_endpoint = # your keys,  
            api_key= # your keys,  
            api_version=# your keys,  
            )
```

### Download Models 
Locate all downloaded models in the  `./checkpoints` directory! The code structure will like below: 
```bash
./checkpoints
    ├── blip2-opt-2.7b
    ├── t2v-turbo 
    │   ├── unet_lora.pt
    │   ├── inference_t2v_512_v2.0.yaml     # downloaded from T2V-turbo official repo 
    ├── VideoCrafter
    │   ├── model.ckpt
    ├── ssam
    │   ├── swinl_only_sam_many2many.pth
```

You can download pre-trained models as below: 
- [T2V-turbo](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt)
- [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
- [MolmoE-1B-0924](https://huggingface.co/allenai/MolmoE-1B-0924)
- [Semantic-SAM (L)](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth)
- BLIP-BLUE for Video Ranking 
```shell
git lfs install
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
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
                        --load_molmo \
                        --selection_score='dsg_blip' \    # video ranking metric 
                        --seed=123 \                      # random seed 
                        --round=1 \                       # iteration round 
                        --k=10 \                          # number of video candidates 
                        --div_seeds                       # use diverse seed per iterative rounds. 
done
```


## 📝 TODO List
- [ ] Release the whole code.


## 📚 BibTeX

💗 If you enjoy our VideoRepair and find some beneficial things, citing our paper would be the best support for us! 

```
@misc{lee2024videorepair,
      title={VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement}, 
      author={Daeun Lee and Jaehong Yoon and Jaemin Cho and Mohit Bansal},
      year={2024},
      eprint={2404.xxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


