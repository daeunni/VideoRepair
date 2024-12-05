import cv2, base64, os, io, re
from PIL import Image
import numpy as np 
from utils_client import ask_gpt4o_DSG_and_grounding_wo_vprompt

def encode_video(video_path) : 
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames


def extract_first_frame(video_path, cur_gen_dir) : 
    base64Frames = encode_video(video_path)[0]          
    first_frame_img = os.path.join(cur_gen_dir, 'first_frame.jpg')
    img_data = base64.b64decode(base64Frames)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(first_frame_img, img)

    img = Image.open(first_frame_img)       
    return img 

def encode_gpt4_input(pil_image_object) : 
    buffered = io.BytesIO() 
    pil_image_object.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8") 
    return base64_image


def key_object_extraction(qid2tuple, qid2question) : 
    object_string = qid2tuple['custom_0']['output']
    lines = object_string.split("\n")
    Q_type = [] ; key_objects_from_Q = []

    # Question type stacking 
    for line in lines:
        category = line.split('|')[1].split('-')[0].strip()
        Q_type.append(category)
        
        if category in ['other', 'entity']:
            match = re.search(r'\(([^)]+)\)', line)
            if match:
                nouns = match.group(1).split(',')[0].strip()  
                key_objects_from_Q.append(nouns)

    key_objects_in_questions = []
    for q in qid2question.values():
        is_key = False
        for obj in key_objects_from_Q:
            if obj in q:
                key_objects_in_questions.append(obj)
                is_key = True 

        if not is_key : 
            key_objects_in_questions.append(None)

    return Q_type, key_objects_from_Q, key_objects_in_questions



def keep_object_number(qid2question, object_wise_dict, preserve_object) : 

    count_priority = None 

    preserve_questions = [item['Q'] for item in object_wise_dict.get(preserve_object, [])]
    local_questions = [q for q in qid2question.values() if q not in preserve_questions]

    try : 
        q_logs = object_wise_dict[preserve_object]
    except : 
        return None, None     

    for log in q_logs :  

        # count question 
        if (len(log) > 2) : 
            # Yes-answered 
            if (log['obj_in_prompt'] == log['obj_in_img'])  :          
                preserve_num = log['obj_in_img']
                count_priority = 1 
                print(f'* [Priority 1] Preserved object : {preserve_object} | Preserved num : {preserve_num}')

            # No-answered 
            elif (log['obj_in_prompt'] < log['obj_in_img']) :        
                preserve_num = log['obj_in_prompt'] 
                count_priority = 2
                print(f'* [Priority 2] Preserved object : {preserve_object} | Preserved num : {preserve_num}')

            elif (log['obj_in_prompt'] > log['obj_in_img']) and (log['obj_in_img'] > 0) :  
                preserve_num = log['obj_in_img'] 
                lack_num_of_object = log['obj_in_prompt'] -  log['obj_in_img'] 
                local_questions.append(f'{lack_num_of_object} {preserve_object}')
                count_priority = 3 

                print(f'* [Priority 3] Preserved object : {preserve_object} | Preserved num : {preserve_num}')
                print(f'* {lack_num_of_object} number of object {preserve_object} need to generate more')

            else : 
                local_questions += preserve_questions
                count_priority = 4 

    return preserve_num, count_priority, preserve_questions, local_questions


def automatic_scoring_w_dsg(videos, cur_gen_dir, qid2question, init_prompt, qid2dependency): 
    all_dsg_scores = []

    ## Candidate video evaluation 
    for video_path in videos : 
        first_frame_img = extract_first_frame(os.path.join(cur_gen_dir, video_path), cur_gen_dir)  
        first_frame_img_gpt = encode_gpt4_input(first_frame_img)
        dsg_answers = ask_gpt4o_DSG_and_grounding_wo_vprompt(first_frame_img_gpt, qid2question, init_prompt)
        qid2scores = {} ; qid2validity = {}
        if dsg_answers == None : 
            continue 

        try : 
            for idx, qa in enumerate(dsg_answers) : 
                qid2scores[str(idx+1)] = qa['A']            # e.g., {'1': 0.0, '2': 0.0, '3': 1.0, '4': 1.0}

            # consider dependency -> modify dsg_answers 
                for id, parent_ids in qid2dependency.items() : 
                    any_parent_answered_no = False

                    for parent_id in parent_ids:
                        if parent_id == 0:              
                            continue 
                        if qid2scores[str(parent_id)] == 0:
                            any_parent_answered_no = True 
                            break 
                    
                    if any_parent_answered_no : 
                        qid2scores[id] = 0.0  
                        try : 
                            dsg_answers[int(id)-1]['A'] = 0.0        
                        except : 
                            continue            
                        qid2validity[id] = False                
                    else :  
                        qid2validity[id] = True                
        except : 
            dsg_answers = dsg_answers

        dsg_score = sum(float(qa['A']) for qa in dsg_answers) / len(dsg_answers)
        all_dsg_scores.append(dsg_score)    

        print('=' * 50)
        print('Video path: ', video_path)
        print('DSG score: ', dsg_score)
        print('=' * 50)

    return all_dsg_scores       



def concatenate_video_1st_frames(cur_gen_dir, video_paths, output_path):
    frames = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(os.path.join(cur_gen_dir, video_path+'.mp4'))
        ret, frame = cap.read()  
        if ret:
            frames.append(frame) 
        cap.release()  

    frame_heights = [frame.shape[0] for frame in frames]
    max_height = max(frame_heights)

    padded_frames = []
    for frame in frames:
        h, w, _ = frame.shape
        if h < max_height:  
            padding = np.zeros((max_height - h, w, 3), dtype=np.uint8)
            frame = np.vstack((frame, padding))
        padded_frames.append(frame)

    concatenated_image = np.hstack(padded_frames)
    cv2.imwrite(os.path.join(cur_gen_dir, output_path), concatenated_image)
    print('Saved 1st frames to ' + os.path.join(cur_gen_dir, output_path))