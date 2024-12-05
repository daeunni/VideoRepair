# Basics 
import argparse, os, time, gc, re, json, random
import warnings;warnings.filterwarnings('ignore')
import torch
import numpy as np
from PIL import Image
from functools import reduce
from lightning_fabric import seed_everything
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM

# Our utils 
from utils_t2v import *
from utils_client import * 
from utils_model import * 
from utils_other import * 
     

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description="VideoRepair")
    parser.add_argument("--eval_section", type=str, default='count', help="evalcrafter section=[action, amp, color, count, face, none, text]")
    parser.add_argument("--model", type=str, default='t2vturbo', help="t2v generation model")
    parser.add_argument("--output_root", type=str, default='/nas-ssd2/daeun/FixYourVideo/results/poc/COUNT/', help="save path root")
    parser.add_argument("--seed", type=int, default=123, help="generation seed")     
    parser.add_argument("--data", default ="evalcrafter", help="benchmark") 
    parser.add_argument("--selection_score", default ="dsg_blip", help="segmentation models") 
    parser.add_argument("--round", type=int, default=1, help="iteration round")
    parser.add_argument("--k", type=int, default=5, help="# of video candidates")

    args = parser.parse_args()

    EVAL_SECTION = args.eval_section
    print('< ', EVAL_SECTION, ' section is start!! >')
    SEED = args.seed 
    ROUND_NUM = args.round 

    ## Load dsg questions (previously saved)
    if args.data == 'evalcrafter' : 
        dsg_json_path = './datasets/evalcrafter/dsg_' + EVAL_SECTION + '.json'   
    elif args.data == 'compbench' : 
        dsg_json_path = './datasets/compbench/' + EVAL_SECTION + '.json'   
    with open(dsg_json_path, 'r') as json_file:
        meta_data = json.load(json_file)

    ## For score selection 
    if 'blip' in args.selection_score : 
        device_blip = torch.device('cuda')
        blip2_processor = AutoProcessor.from_pretrained("./checkpoints/blip2-opt-2.7b")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("./checkpoints/blip2-opt-2.7b", torch_dtype=torch.float16).to(device_blip)#.to('cuda')

    ## Root setting 
    cur_result_root = os.path.join(args.output_root, EVAL_SECTION) + '/'
    print(cur_result_root)
    if not os.path.exists(cur_result_root):  
        os.makedirs(cur_result_root)

    ## Load T2V-turbo 
    if args.model == 't2vturbo' :      
        pipeline, multidiffusion_pipeline = load_t2vturbo(SEED)

    ## Load Molmo 
    processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
    )
    molmo = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    cur_times_zip = []
    start_time = time.time()


    for INIT_PROMPT_IDX in range(len(meta_data) ) :     

        torch.manual_seed(SEED)
        seed_everything(SEED)  

        cur_meta = meta_data[INIT_PROMPT_IDX]              
        init_prompt = cur_meta['origin_prompt']              
        unique_id = cur_meta['idx']
        del cur_meta["origin_prompt"]
        del cur_meta["idx"]
        qid2question = cur_meta['qid2question']
        qid2tuple = cur_meta['qid2tuple']
        qid2dependency = cur_meta['qid2dependency']           
        print('< Initial prompt > : ', init_prompt)

        cur_gen_best_dir  = os.path.join(cur_result_root, str(unique_id), 'best_videos_' + args.selection_score)   
        os.makedirs(cur_gen_best_dir, exist_ok=True) 

        # if there are DSG answers, pass 
        if os.path.exists(os.path.join(cur_gen_best_dir, 'dsg_log.txt')) :    
            print(str(unique_id), ' is already existed')
            continue 
            
        cur_start_time = time.time()
        
        for cur_round in range(ROUND_NUM) : 

            print(f'{cur_round} round is started')

            seed_fields = [random.randint(0, 100000) for _ in range(args.k * ROUND_NUM)] 
            SEED_LISTS = seed_fields[args.k * cur_round : args.k * (cur_round+1)]
            print('SEED_LISTS : ', SEED_LISTS)
            
            cur_gen_dir  = os.path.join(cur_result_root, str(unique_id), str(cur_round) + '_round')    # directory per prompts  
            os.makedirs(cur_gen_dir, exist_ok=True) 

            # if first round 
            if cur_round == 0 : 
                init_video_path = os.path.join(cur_gen_dir, init_prompt + '_0.mp4')
                noise_path = None 
            else : 
                init_video_path = os.listdir(cur_gen_dir)[0]
                noise_seed = re.search(r'_(\d+)\.mp4$', init_video_path).group(1)
                prev_gen_dir = cur_gen_dir.replace(str(cur_round)+ '_round', str(cur_round-1)+ '_round')
                noise_path = os.path.join( prev_gen_dir,  'init_latent_' + str(noise_seed) + '.pt')

            # If there are DSG answers, pass 
            if os.path.exists(os.path.join(cur_gen_dir, 'binary_yes_mask.png')) :    
                print(str(unique_id), ' is already existed')
                continue 
            
            if cur_round == 0 : 
                print(f'---------------- [ 1. Initial generation ]----------------')
                if not os.path.exists(init_video_path) :        
                    if args.model == 't2vturbo' : 
                        T2VTurbo_from_each_prompt(outpath = cur_gen_dir, 
                                                                        prompt_text = init_prompt, 
                                                                        seeds = SEED, pipeline = pipeline)      
                    elif args.model == 'videocrafter2' : 
                        with open(os.path.join(cur_gen_dir, 'init_prompt.txt'), 'w', encoding='utf-8') as file:  
                            file.write(init_prompt)
                        VideoCrafter_from_each_prompt(outpath = cur_gen_dir, 
                                                        prompts_path = os.path.join(cur_gen_dir, 'init_prompt.txt'), 
                                                        seeds = SEED
                                                        )

            print(f'---------------- [ 2. Video Evaluation ] ----------------')    

            # key-object extraction 
            Q_type, key_objects_from_Q, key_objects_in_questions = key_object_extraction(qid2tuple, qid2question)
            print('Question types: ', Q_type)
            print('Key objects: ', key_objects_from_Q)

            # DSG question asking 
            if cur_round == 0 : 
                video_path = init_video_path
            else : 
                video_path = os.path.join(cur_gen_dir, init_video_path )

            first_frame_img = extract_first_frame(video_path, cur_gen_dir)  
            first_frame_img_gpt = encode_gpt4_input(first_frame_img)
            
            dsg_answers = video_evaluation(qid2question, 
                                            first_frame_img_gpt, 
                                            Q_type, 
                                            key_objects_in_questions)

            dsg_score = sum(float(qa['A']) for qa in dsg_answers) / len(dsg_answers)

            # DSG== 1.0 -> pass 
            if dsg_score == 1.0 :          
                break 

            # Write dsg answers 
            f = open(os.path.join(cur_gen_dir, 'logs.txt'), 'w')             
            f.write('\n'.join(list(map(str, dsg_answers))))
            f.close()

    
            print(f'---------------- [ 3. Refinement planning ] ----------------') 
            preserve_object = None ; preserve_num = None      
            preserve_prompts = [] ; local_prompts = []
            
            # Q1. Which object we should preserve? 
            preserve_object, object_wise_dict = keep_object_selection(key_objects_from_Q, dsg_answers, first_frame_img_gpt)

            # Q2. How many object we should preserve / add / delete? 
            preserve_num, count_priority, preserve_questions, local_questions = keep_object_number(qid2question, object_wise_dict, preserve_object)
            if preserve_num == None : 
                continue 

            # Q3. How to localize which object need to delete? 
            pointing_prompt = None 
            if preserve_object != None : 
                if preserve_num != None : 
                    pointing_prompt = f'Point the biggest {preserve_num} {preserve_object}.'
                else : 
                    pointing_prompt = f'Point the biggest {preserve_object}.'
                    
            # Logging 
            output = '\n' + '=' * 50 + '\n' 
            output += f'* [DSG score]: {dsg_score}\n'
            output += f'* [Object decision] Preserved object : {preserve_object} | Preserved num : {preserve_num} | Priority: {count_priority}\n'
            output += f'* [Pointing prompt]: {pointing_prompt}\n'
            output += f'* [DSG questions for preserving prompts]: [{", ".join(preserve_questions)}]\n'
            output += f'* [DSG questions for local prompts]: [{", ".join(local_questions)}]\n'
            output += '=' * 50 + '\n'
            file_path = os.path.join(cur_gen_dir, 'logs.txt')
            with open(file_path, 'a') as f:
                f.write(output)
            print(output)


            print(f'---------------- [ 4. Pointing and Mask generation ] ----------------') 
            remaining_object = [i for i in key_objects_from_Q if i != preserve_object]

            # Molmo pointing 
            viz_path = os.path.join(cur_gen_dir, 'molmo_point.png')
            object_points_list = ask_molmo(processor, molmo, first_frame_img, pointing_prompt, viz_path)     # Ask Molmo pointing 
            print('* Object point: ', object_points_list)

            # Semantic-SAM 
            if len(object_points_list) > 0 and (len(local_questions) != 0) and (len(remaining_object) != 0) :    
                point2mask_semanticsam(img_path=os.path.join(cur_gen_dir,'first_frame.jpg'), 
                                        point_lists=object_points_list, 
                                        mask_save_path=os.path.join(cur_gen_dir, 'ssam_mask_stack.npy'), 
                                        img_width=512, img_height=320)
                all_masks = np.load(os.path.join(cur_gen_dir, 'ssam_mask_stack.npy'))
                total_mask = reduce(np.logical_or, all_masks).squeeze()    
                total_binary_mask = np.where(total_mask, 0, 255).astype(np.uint8)    

            else : 
                total_binary_mask = np.full((320, 512), 255, dtype=np.uint8)   
                local_questions = preserve_questions + local_questions         

            mask_paths = [os.path.join(cur_gen_dir, 'binary_yes_mask.png')]
            total_binary_mask = Image.fromarray(total_binary_mask)
            total_binary_mask.save(mask_paths[0])


            print(f'---------------- [ 5. Background / Local prompt generation ]----------------') 

            if (len(object_points_list) == 0) or len(remaining_object) == 0 or len(local_questions) == 0 :    
                change_prompt_white = paraphrasing_prompt(init_prompt)[0]
            else : 
                if count_priority == 3 :      
                    change_prompt_white = prompt_generator_from_Q(question_list = local_questions) 

                else : 
                    change_prompt_white = prompt_generator_from_Q(question_list = local_questions) 

            t = '=' * 50 + '\n'
            t += f'* Regenerating prompt : {change_prompt_white}\n'
            t += '=' * 50 + '\n'
            print(t)

            file_path = os.path.join(cur_gen_dir, 'logs.txt')
            with open(file_path, 'a') as f:
                f.write(t)

            print(f'---------------- [ 6. Video Regeneration ] ----------------') 
   
            all_prompt_origin = [init_prompt, change_prompt_white]    

            # filename length limits 
            if len(change_prompt_white) > 30:          
                suffix=""
            elif change_prompt_white == 'None' :         
                change_prompt_white = init_prompt
            else :  
                suffix = change_prompt_white

            file_names = []

            for ith_video, new_seed in enumerate(SEED_LISTS) : 
                print(f'{ith_video} th video generation')
                cur_file = init_prompt + '_' + suffix+'_'+ str(new_seed)
                file_names.append(cur_file)
                
                if args.model == 't2vturbo' : 
                    T2VTurbo_refinement(outpath = cur_gen_dir, 
                                        round_num = cur_round, 
                                        noise_map = noise_path, 
                                        prompt_text = init_prompt,           
                                        seeds = SEED, 
                                        pipeline = multidiffusion_pipeline, 
                                        all_prompt = all_prompt_origin,    
                                        local_seed = new_seed, 
                                        mask_path = mask_paths,                   
                                        suffix = suffix+'_'+ str(new_seed)     
                                        )

                elif args.model == 'videocrafter2' : 
                    prompts_path = os.path.join(cur_gen_dir, 'all_prompts.txt')
                    with open(prompts_path, "w") as file:
                        for prompt in all_prompt_origin:
                            file.write(f"{prompt}\n")

                    VideoCrafter_refinement(outpath = cur_gen_dir, 
                                            all_prompt = prompts_path,    
                                            mask_path = mask_paths[0], 
                                            seeds = SEED, 
                                            )

            concatenate_video_1st_frames(cur_gen_dir=cur_gen_dir, video_paths=file_names, output_path='whole_' + suffix + '.png')


            print(f'---------------- [ 7. Video Ranking ] ----------------') 
            video_ranking(cur_gen_dir, 
                        cur_gen_best_dir, 
                        args.selection_score, 
                        cur_round, 
                        qid2question, 
                        init_prompt, 
                        qid2dependency,
                        blip2_model, 
                        blip2_processor)

            # memory clean 
            gc.collect()
            torch.cuda.empty_cache()
        
        cur_end_time = time.time()
        cur_time_seconds = cur_end_time - cur_start_time
        cur_times_zip.append(cur_time_seconds)
        print('Cur_time_second: ', cur_time_seconds)
        with open(os.path.join(cur_result_root, 'ours_timeconsump.txt'), "w") as file:
            for item in cur_times_zip:
                file.write(f"{item}\n")  


    print('Process are done!')

    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    hours = int(total_time_minutes // 60) ; minutes = int(total_time_minutes % 60)
    print(f"Total Execution time: {hours} h {minutes} min")
