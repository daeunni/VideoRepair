import torch, re, os, shutil
import cv2, subprocess
import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from transformers import GenerationConfig
import matplotlib.pyplot as plt

from utils_other import automatic_scoring_w_dsg


def point2mask_semanticsam(img_path, point_lists, mask_save_path, img_width=512, img_height=320) : 
    command = [
        "python3", "./SemanticSAM/ssam.py",    
        "--img_path", img_path,
        "--point_lists", str(point_lists),
        "--mask_save_path", mask_save_path, 
    ]
    print(command)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Command executed successfully with output:")
        print(result.stdout)
    else:
        print("Command execution failed with error:")
        print(result.stderr)


def ask_molmo(processor, molmo, PIL_input_img, input_prompt, viz_path, ori_w=512, ori_h=320) : 
    # process the image and text
    inputs = processor.process(images=[PIL_input_img], text=input_prompt,)
    original_height, original_width = ori_h, ori_w

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(molmo.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = molmo.generate_from_batch(
                                    inputs,
                                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                                    tokenizer=processor.tokenizer
                                )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    string = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(string)
 
    try :            # if point is >2
        x = float(re.search(r'x="([0-9.]+)"', string).group(1))  
        y = float(re.search(r'y="([0-9.]+)"', string).group(1)) 
        points_list = [[x, y]]
    except :         # if point is 1
        coordinates = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', string)
        points_list = [[float(x_value), float(y_value)] for _, x_value, _, y_value in coordinates]

    # transform to pixel axis 
    pixel_points = []
    for output_coordinates in points_list : 
        X_pixel = (original_width * output_coordinates[0]) / 100
        Y_pixel = (original_height * output_coordinates[1]) / 100   
        pixel_points.append([X_pixel, Y_pixel])

    # visualize 
    background = np.array(PIL_input_img)
    plt.figure(figsize=(10, 6))
    plt.imshow(background)
    for point in pixel_points:
        plt.scatter(point[0], point[1], color='white', s=100, edgecolor='blue', linewidths=2)  
        plt.text(point[0] + 5, point[1] - 5, f'({point[0]:.2f}, {point[1]:.2f})', color='white', fontsize=12)
    plt.axis('off') 
    plt.savefig(viz_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()
    return pixel_points


def compute_max(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)

    

def calculate_blip_bleu(video_path, original_text, blip2_model, blip2_processor):
    cap = cv2.VideoCapture(video_path)
    scorer_cider = Cider()
    bleu1 = Bleu(n=1)
    bleu2 = Bleu(n=2)
    bleu3 = Bleu(n=3)
    bleu4 = Bleu(n=4)

    # Extract frames from the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
    # Get five captions for one video
    Num = 5
    captions = []
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    for i in range(Num):
        frame = extracted_frames[i]
        inputs = blip2_processor(images=frame, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = blip2_model.generate(**inputs)
        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)


    original_text = [original_text]
    cider_score = (compute_max(scorer_cider, original_text, captions))
    bleu1_score = (compute_max(bleu1, original_text, captions))
    bleu2_score = (compute_max(bleu2, original_text, captions))
    bleu3_score = (compute_max(bleu3, original_text, captions))
    bleu4_score = (compute_max(bleu4, original_text, captions))

    blip_bleu_caps_avg = (bleu1_score + bleu2_score + bleu3_score + bleu4_score)/4
     
    return blip_bleu_caps_avg


def video_ranking(cur_gen_dir, 
                        cur_gen_best_dir, 
                        selection_score, 
                        cur_round, 
                        qid2question, 
                        init_prompt, 
                        qid2dependency,
                        blip2_model, 
                        blip2_processor): 

    if selection_score == 'dsg_blip' : 
        
        videos = sorted([f for f in os.listdir(cur_gen_dir) if f.endswith('.mp4')])
        # filtered_videos = sorted([file for file in videos if init_prompt in file])        

        # if args.not_initprompt_for_background : 
        filtered_videos = sorted([file for file in videos if init_prompt not in file])  
        filtered_videos.append(videos[0])

        candidates_dsg_scores = automatic_scoring_w_dsg(filtered_videos, cur_gen_dir, qid2question, init_prompt, qid2dependency)
        max_value = max(candidates_dsg_scores)  
        max_indices = [i for i, score in enumerate(candidates_dsg_scores) if score == max_value]


        # select using blip 
        if len(max_indices) == 1 : 
            max_index = max_indices[0]
        else : 
            max_clip_score = 0 ; max_index = None 
            for idx in max_indices : 
                cur_video_path = os.path.join(cur_gen_dir, filtered_videos[idx])
                score = calculate_blip_bleu(cur_video_path, init_prompt, blip2_model, blip2_processor)
                print(f'{idx} video clip score: {score}')
                if score > max_clip_score : 
                    max_clip_score = score
                    max_index = idx 
    
        max_score_file = filtered_videos[max_index]
        full_max_score_file = os.path.join(cur_gen_dir, max_score_file)
        print(full_max_score_file)

        # copy next round dir 
        next_cur_gen_dir = cur_gen_dir.replace(str(cur_round) + '_round', str(cur_round+1) + '_round')
        os.makedirs(next_cur_gen_dir, exist_ok=True) 
        shutil.copy(full_max_score_file, os.path.join(next_cur_gen_dir, max_score_file  ))
        shutil.copy(full_max_score_file, os.path.join(cur_gen_best_dir, str(cur_round) + '_best_video.mp4'   ))
