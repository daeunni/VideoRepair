import configparser, time, re
from openai import AzureOpenAI

config = configparser.ConfigParser()
config.read('config.ini')

# Set up your own OpenAI API 
client = AzureOpenAI(
            azure_endpoint = config.get("openai", "azure_endpoint"), 
            api_key= config.get("openai", "api_key"), 
            api_version= config.get("openai", "api_version"), 
            )


def asking_gpt4o(system_prompt, task_prompt, gpt4_input_image) : 
    response = client.chat.completions.create( 
                                model="gpt-4o",           # "gpt-4o-new"
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text":  task_prompt}, 
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gpt4_input_image}"}},
                                        ],
                                    }
                                ],
                                max_tokens=100,
                            )
    answer = response.choices[0].message.content  
    return answer 


def filter_DSG_answer_w_dependency(dsg_answers, qid2dependency) : 
    qid2scores = {} ; qid2validity = {}

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
    
    return qid2scores, qid2validity, dsg_answers



def ask_gpt4o_DSG_and_grounding_wo_vprompt(gpt4_input_image, qid2question, init_prompt) : 
    dsg_answers_with_area = []
    for i in range(len(qid2question)) : 
        cur_question = qid2question[str(i+1)]              
        system_prompt = f'You are an expert at answering questions about the content of a given image.'

        task_prompt = f'1. Given the question: "{cur_question}", provide a brief reasoning (up to two sentences) to determine an accurate answer. \
                        2. Respond using binary values: 1.0 for Yes and 0.0 for No. If the answer is uncertain due to image distortion or other issues, respond with 0.0 (No). \
                        Return the result as a dictionary in the following format (not in JSON format): \
                        {{"Q": "<question>", "reasoning": "<brief reasoning>", "A": <binary answer>}} \
                        (e.g., {{"Q": "Is there one robot?", "reasoning": "There are two visible robots in the image. To guarantee a Yes answer, one robot should be removed.", "A": 0.0}}) \
                        Provide only the dictionary as the output, without any additional text or explanations.'

        success = False ; error_count = 0 
        while not success:
            try : 
                response = client.chat.completions.create( 
                                                model="gpt-4o",           
                                                messages=[
                                                    {"role": "system", "content": system_prompt},
                                                   {
                                                        "role": "user",
                                                        "content": [
                                                            {"type": "text", "text":  task_prompt}, 
                                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gpt4_input_image}"}},
                                                        ],
                                                    }
                                                ],
                                                max_tokens=100,
                                            )

                answer = response.choices[0].message.content  
                print(answer)
                print('*' * 5)
                success = True 

            except : 
                print('ERROR..')
                time.sleep(9)
                error_count += 1 
                if error_count > 3 :          
                    return  
        try : 
            answer = answer.replace('"', '\\"')
            answer_dict = eval(answer.replace('\n', '').replace('```json', '').replace('```', '').replace('\\', '').replace('```python', ''))
        except : 
            return 
        dsg_answers_with_area.append(answer_dict)
    return dsg_answers_with_area


def video_evaluation(qid2question, first_frame_img_gpt, Q_type, key_objects_in_questions) : 
    dsg_answers = []
    for i in range(len(qid2question)) : 
        cur_question = qid2question[str(i+1)]   
        cur_question_type = Q_type[i]
        key_objects = key_objects_in_questions[i]          

        system_prompt = f'You are an expert at answering questions about the content of a given image.'

        # Devide count prompt & non-count prompt 
        count_prompt = f'''
                        1. Given the question: "{cur_question}", provide a brief reasoning (up to two sentences) to determine the accurate answer.
                        2. Respond to the question using binary values: 1.0 for "Yes" and 0.0 for "No". If the answer is uncertain or unnatural due to image distortion or other issues, respond with 0.0 ("No").
                        3. Return the number of "{key_objects}" (as an integer) mentioned in the initial prompt "{cur_question}". 
                        4. Return the number of "{key_objects}" (as an integer) in the provided image.

                        Return the result as a dictionary in the following format (not in JSON format):
                        {{
                            "Q": "<question>",
                            "A": <binary answer>,
                            "reasoning": "<brief reasoning>",
                            "obj_in_prompt": <number of key object mentioned in the initial prompt>,
                            "obj_in_img": <number of key object in the image>,
                        }}

                        Example: 
                        {{
                            "Q": "Is there one robot?",
                            "A": 0.0,
                            "reasoning": "There are two visible robots in the image.",
                            "obj_in_prompt": 1,
                            "obj_in_img": 2,
                        }}

                        Please provide only the dictionary as the output without any additional text or explanation.
                        '''

        non_count_prompt = f'''
                            Respond to "{cur_question}" using binary values: 1.0 for Yes and 0.0 for No. If the answer is uncertain due to image distortion or other issues, respond with 0.0 (No). \
                            Return the result as a dictionary in the following format (not in JSON format): \        
                            {{"Q": "<question>", "A": <binary answer>}} \
                            (e.g., {{"Q": "Is there one robot?", "A": 0.0}}) \
                            Provide only the dictionary as the output, without any additional text or explanations.
                            '''    

        success = False
        while not success:
            try : 
                if cur_question_type == 'other' :          
                    answer = asking_gpt4o(system_prompt, count_prompt, first_frame_img_gpt)
                else : 
                    answer = asking_gpt4o(system_prompt, non_count_prompt, first_frame_img_gpt)

                print(answer) ; print('*' * 5)
                success = True 

            except : 
                print('ERROR..')
                time.sleep(9)

        try : 
            answer = answer.replace('"', '\\"')
            answer_dict = eval(answer.replace('\n', '').replace('```json', '').replace('```', '').replace('\\', '').replace('```python', ''))
        except : 
            continue 

        dsg_answers.append(answer_dict)
        
    try : 
        qid2scores, qid2validity, dsg_answers = filter_DSG_answer_w_dependency(dsg_answers, qid2dependency) 
        print('Updated DSG score: ', qid2scores)        
        print('Updated logs: ', qid2validity)
    except : 
        dsg_answers = dsg_answers         # error -> not consider dependency 

    return dsg_answers



def keep_object_selection(key_objects_from_Q, dsg_answers, first_frame_img_gpt) : 
    preserve_object = None

    # Object-wise question collection 
    object_wise_dict = {}
    for obj in key_objects_from_Q : 
        cur_obj_qas = []
        for cur_qa in dsg_answers:
            if (obj in cur_qa['Q']) or (re.search(r'\b' + r'\b|\b'.join(obj.split()) + r'\b', cur_qa['Q'], re.IGNORECASE)) :     
                cur_obj_qas.append(cur_qa)
        object_wise_dict[obj] = cur_obj_qas


    task_prompt_key = (
        f"Given the generated image and the list of question-answer pairs for each object, represented as {object_wise_dict}, "
        "choose the most accurately or visibly generated object from the list {key_objects_from_Q}. "
        "Prioritize selecting objects with a high number of answers rated 1.0 for each question."
        "Select the object that is both large and clearly visible, prioritizing prominent objects (such as animals, humans, or specific items) over background elements (like ocean or city). "
        "Return only the name of the best object to keep from the list, without additional explanation (e.g., 'dog')."
    )

    stop = False ; error_count = 0 
    while not stop:
        try : 
            local_response = client.chat.completions.create( 
                                            model="gpt-4o",    
                                            messages=[
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {"type": "text", "text":  task_prompt_key}, 
                                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_frame_img_gpt}"}},
                                                    ],
                                                }
                                            ],
                                            max_tokens=100,
                                        )
            preserve_object = local_response.choices[0].message.content  
            stop = True 

        except : 
            print('ERROR..')
            time.sleep(9)
            error_count += 1 
            if error_count > 3 :          # stop condition 
                preserve_object = None 
                stop = True   
    
    return preserve_object, object_wise_dict


def paraphrasing_prompt(origin_prompt) : 
        task_prompt_key = (
            f"Given the prompt: {origin_prompt}, generate 1 paraphrases of the initial prompt which keep the semantic meaning."
            "Respond with each new prompt in between <PROMPT> and </PROMPT>, eg: <PROMPT>paraphrase </PROMPT>. Answer using a single phrase. Do NOT generate any explanation, write only answer."
        )

        stop = False ; error_count = 0 
        while not stop:
            try : 
                new_prompt = client.chat.completions.create( 
                                                model="gpt-4-0125",
                                                messages=[
                                                    {
                                                        "role": "user",
                                                        "content": [
                                                            {"type": "text", "text":  task_prompt_key}, 
                                                        ],
                                                    }
                                                ],
                                                max_tokens=100,
                                            )
                new_prompt = new_prompt.choices[0].message.content  
                new_prompt = re.findall(r'<PROMPT>(.*?)</PROMPT>', new_prompt)
                stop = True 

            except : 
                print('ERROR..')
                time.sleep(9)
                error_count += 1 
                if error_count > 3 :           
                    local_prompt_answer = None 
                    stop = True 

            return new_prompt  


def prompt_generator_from_Q(question_list) : 

    system_prompt_local = (
        "You are an expert in rephrasing prompts for a text-to-video model based on the given questions."
    )

    task_prompt_local = (
        f"Given the following list of questions {question_list}, \
        create a single descriptive sentence that combines the meaning of each question into a natural, affirmative statement that provides a full, concise summary."
        "Your response should be a concise 1 phrase, without additional explanation.  (e.g., 'a small bear')"
        "Examples: "
    )

    examples = """

        - Example 1 
            Question list: ['Is there a bed?', 'Is the bed blue?', 'Are the pillows beige?', 'Are the pillows with the bed?']
            Answer: "Blue bed with beige pillows."

        - Example 2 
            Question list: [Are there three real bears?]
            Answer: "Three real bears."

        - Example 3 
            Question list: [Are there two people?, Are the people making pizza?]
            Answer: "Two people making pizza.

        - Example 4 
            Question list: [Is there a family?, Is there one cat?, Is there a park?, Is the family taking a walk?, Is the cat walking?, Is the family enjoying?, Is the family breathing fresh air?, Is the family exercising?]
            Answer: "A family and a cat are walking in the park."

        - Example 5 
            Question list: [Is there a green bench?, Is there an orange tree?, Is the bench green?, Is the tree orange?]
            Answer: "Green bench and orange tree."

    Your Current Task: Your response should be a concise 1 phrase, without additional explanation (e.g., "a small bear")

    """

    stop = False ; error_count = 0 
    while not stop:
        try : 
            local_response = client.chat.completions.create( 
                                            model="gpt-4-0125",
                                            messages=[
                                                {"role": "system", "content": system_prompt_local},
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {"type": "text", "text":  task_prompt_local + examples}, 
                                                    ],
                                                }
                                            ],
                                            max_tokens=100,
                                        )
            local_prompt_answer = local_response.choices[0].message.content  
            stop = True 

        except : 
            print('ERROR..')
            time.sleep(9)
            error_count += 1 
            if error_count > 3 :           
                local_prompt_answer = None 
                stop = True   
    return local_prompt_answer

