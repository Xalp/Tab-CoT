import argparse
import logging
import torch
import random
import time
import os
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
    total = 0
    correct_list = []    

    for i, data in enumerate(dataloader):
        if i + 1 < args.start:
            continue

        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        # x = "Q: " + x[0] + "\n" + "A:"
        x = x[0]
        y = y[0].strip()
        
        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot" and args.self_consistency==False:
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x + args.cot_trigger
        else:
            x = x

        if args.self_consistency:
            preds = []
            cot_triggers = ["\n\n|step|subquestion|process|result|","\n\n|step|question|response|", "\n\n|step|subquestion|procedure|result|", "\n\n|step|question|process|result|", "\n\n|step|question|procedure|result|", "\n\n|step|subquestion|response|"]
            for cot_trigger in cot_triggers:
                x2 = x + " " + cot_trigger
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                pred = decoder.decode(args, x2, max_length, i, 1)
                z2 = x2 + pred + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length, i, 2)
                # print(z2 + pred)
                pred = answer_cleansing(args, pred)
                if pred is not None and pred != "":
                    preds.append(pred)
            # print(preds)
            if len(preds) != 0:
                pred = max(preds, key=preds.count)
            else:
                pred = ""
            

        else: 
            if args.method=="few_shot_cot":
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                pred = decoder.decode(args, x, max_length, i, 1)
                print(x + pred)
                pred = answer_cleansing(args, pred)
                if args.dataset == "last_letters":
                    pred = pred[-4:]
            elif args.method=="few_shot":
                x = "Q: " + x + "\n" + "A:"
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                pred = decoder.decode(args, x, max_length, i, 1)
                print(x + pred)
                pred = answer_cleansing(args, pred)
            elif args.method=="ltsbs":
                x = "Q: " + x + "\n" + "A: Let's think step by step."
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                pred = decoder.decode(args, x, max_length, i, 1)
                z2 = x + pred + " " + args.direct_answer_trigger_for_zeroshot_cot
                pred = decoder.decode(args, z2, max_length, i, 1)
                print(z2 + pred)
                pred = answer_cleansing(args, pred)
            else:
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                pred = decoder.decode(args, x, max_length, i, 1)
                z2 = x + pred + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct
                pred = decoder.decode(args, z2, max_length, i, 2)
                print(z2 + pred)
                pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
            
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)

        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["ARC_C","aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code", "code-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "ltsbs"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=512, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="For test only, starting position"
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=3.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--self_consistency", action="store_true", help=" Whether to use self-consistency"
        )

    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "ARC_C":
        args.dataset_path = "./dataset/ARC_C/ARC-Challenge-Dev.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = '''|step|subquestion|process|result|'''
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "|step|subquestion|procedure|result|"
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "|step|subquestion|process|result|"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "|step|initial state|action|next state|"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "|step|original answer|action|updated answer|"
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "|step|word|last letter|answer|"
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "|step|original answer|action|updated answer|"
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "|step|initial coin state|flip or not|next coin step|"
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    args.cot_trigger = "\n\n" + args.cot_trigger

    return args

if __name__ == "__main__":
    main()