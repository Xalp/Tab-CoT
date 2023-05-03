from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import openai # For GPT-3 API ...
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import torchtext
import re
import random
import time
import datetime
import pandas as pd
import sys

# https://review-of-my-life.blogspot.com/2017/11/python-dict-shuffle.html
def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)
  
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input, max_length, i, k):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(args.api_time_interval)
    
    # https://beta.openai.com/account/api-keys

    openai.api_key = #put your api here
    # print(openai.api_key)
    
    # Specify engine ...
    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    elif args.model == "code":
        engine = "code-cushman-001"
    elif args.model == "code-xl": # Please change this back
        engine = "code-davinci-002"
    else:
        raise ValueError("model is not properly defined ...")
    cnt = 0
    while True:
        try:
            response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=0,
            stop='\n\n'
            )
            return response["choices"][0]["text"]
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except openai.error.RateLimitError:
            time.sleep(2)
            continue
        except openai.error.ServiceUnavailableError:
            time.sleep(2)
            continue
        except openai.error.APIConnectionError:
            time.sleep(2)
            continue
        except:
            time.sleep(2)
            continue
    

class Decoder():
    def __init__(self, args):
        print_now()
 
    def decode(self, args, input, max_length, i, k):
        response = decoder_for_gpt3(args, input, max_length, i, k)
        return response

def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset in ("commonsensqa","ARC_C"):
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(args.dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "svamp":
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)            
          
    elif args.dataset in ("coin_flip", "last_letters"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=args.minibatch_size,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader

# ver 0.2
def answer_cleansing(args, pred):

    print("pred_before : " + pred)
    
    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa","ARC_C"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    # elif args.dataset in ("gsm8k"):
    #     pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("gsm8k","addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot", "ltsbs"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred

def create_demo_text(args, cot_flag):
    x, z, y = [], [], []

    if args.dataset in ("aqua"):
        return '''John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the new mean?|40 + 10 = 50|50|
Therefore, among A through E, the answer is A.

If a / b = 3/4 and 8a + 5b = 22, then find the value of a. Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2

|step|subquestion|process|result|
|---|---|---|---|
|1|What equation we have have if we substitute b with a?|a / b = 3/4; b = 4a / 3; 8a + 5(4a / 3) = 22|8a + 5(4a / 3) = 22|
|2|What is the value of a?|8a + 5(4a / 3) = 22; 8a + 20a / 3 = 22; 44a / 3 = 22; a = 3/2|2/3|
Therefore, among A through E, the answer is B.

A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the distance this person traveling?|20 * 2.5 = 50|50km|
Therefore, among A through E, the answer is E.

How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788

|step|subquestion|process|result|
|---|---|---|---|
|1|How many one-digit numbers are there?|9-1+1=9|9|
|2|How many two-digit numbers are there?|99-10+1=90|90|
|3|How many three-digit numbers are there?|500-100+1=401|401|
|4|How many keystrokes are needed to type the number from 1 to 500?|9 + 90*2 + 401*3 = 1392|1392|
Therefore, among A through E, the answer is B.

'''


    if args.dataset in ("coin_flip"):
        return '''A coin is heads up. Dorian flips the coin. Mayra flips the coin. Freddie does not flip the coin. Magaly flips the coin. Is the coin still heads up? Note that "flip" here means "reverse".

|step|subquestion|process|result|
|---|---|---|---|
|1|Is the coin heads up?|Dorian flips the coin.|The coin is tails up.|
|2|Is the coin heads up?|Mayra flips the coin.|The coin is heads up.|
|3|Is the coin heads up?|Freddie does not flip the coin.|The coin is heads up.|
|4|Is the coin heads up?|Magaly flips the coin.|The coin is tails up.| 
Therefore, the answer (Yes or No) is "No".\n\n'''

    if args.dataset in ("last_letters"):
        return '''Take the last letters of each words in \"Lucky Mireya Jj Kc\" and concatenate them.

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the last letter of "Lucky"?|"Lucky"[-1] = 'y'|answer = 'y'|
|1|What is the last letter of "Mireya"?|"Mireya"[-1] = 'a'|answer = 'y' + 'a' = 'ya'|
|1|What is the last letter of "Jj"?|"Jj"[-1] = 'j'|answer = 'ya' + 'j' = 'yaj'|
|1|What is the last letter of "Kc"?|"Kc"[-1] = 'c'|answer = 'yaj' + 'c' = 'yajc'|
Therefore, the answer is "yajc".\n\n'''
        
    if args.dataset == "commonsenseqa":
        return '''What do people use to absorb extra ink from a fountain pen? Answer Choices: (A) shirt pocket (B) calligrapher’s hand (C) inkwell (D) desk drawer (E) blotter

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer must be an item that can absorb ink.|(E)|
Therefore, Among A through E, the answer is E.

What home entertainment equipment requires cable? Answer Choices: (A) radio shack (B) substation (C) television (D) cabinet

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer must require cable.|(C)|
Therefore, Among A through E, the answer is C.

The fox walked of city into the forest, what was it looking for? Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer must be something in the forest.|(B)|
Therefore, Among A through E, the answer is B.

Sammy wanted to go to where the people were. Where might he go? Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer must be a place with a lot of people.|(A)|
Therefore, Among A through E, the answer is A.

Where do you put your grapes just before checking out? Answer Choices: (A) mouth (B) grocery cart (C)super market (D) fruit basket (E) fruit market

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer should be the place where grocery items are placed before checking out.|(B)|
Therefore, Among A through E, the answer is B.

Google Maps and other highway and street GPS services have replaced what? Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer must be something that used to do what Google Maps and GPS services do, which is to give directions.|(D)|
Therefore, Among A through E, the answer is D.

Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness

|step|subquestion|process|result|
|---|---|---|---|
|1|What can we know of answer?|The answer should be the feeling of someone getting divorced who was doing all the work.|(C)|
Therefore, Among A through E, the answer is C.

'''
    if args.dataset == "strategyqa":
        return '''Do hamsters provide food for any animals?

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the evidence?|Hamsters are prey animals. Prey are food for predators.|yes|
Therefore, the answer (yes or no) is yes.

Could Brooke Shields succeed at University of Pennsylvania?

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the evidence?|Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.|yes|
Therefore, the answer (yes or no) is yes.

Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the evidence?|Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5.|no|
Therefore, the answer (yes or no) is no.

Yes or no: Is it common to see frost during some college commencements?

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the evidence?|College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.|yes|
Therefore, the answer (yes or no) is yes.

Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?

|step|subquestion|process|result|
|---|---|---|---|
|1|What is the evidence?|The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.|no|
Therefore, the answer (yes or no) is no.

Yes or no: Would a pear sink in water?

|step|subquestion|process|result|
|---|---|---|---|
|1|The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float.|no|
Therefore, the answer (yes or no) is no.

'''
    # example sentences ...    
    if True: #args.dataset in ("multiarith", "gsm8k"):
        return '''There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many trees are in the grove?|15|15|
|2|How many trees will be in the grove after the grove workers are done?|21|21|
|3|How many trees did the grove workers plant today?|21 - 15|6|
Therefore, the answer (arabic numerals) is 6.

If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many cars are in the parking lot?|3|3|
|2|How many cars arrive?|2|2|
|3|How many cars are in the parking lot?|3 + 2|5|
Therefore, the answer (arabic numerals) is 5.

Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many chocolates did Leah have?|32|32|
|2|How many chocolates did her sister have?|42|42|
|3|How many chocolates did they eat?|35|35|
|4|How many chocolates did they eat?|35|35|
|5|How many chocolates do they have left in total?|32 + 42 - 35|39|
Therefore, the answer (arabic numerals) is 39.

Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many lollipops did Jason have?|20|20|
|2|How many lollipops does Jason have now?|12|12|
|3|How many lollipops did Jason give to Denny?|20 - 12|8|
Therefore, the answer (arabic numerals) is 8.

Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many toys does Shawn have?|5|5|
|2|How many toys did he get from his mom?|2|2|
|3|How many toys did he get from his dad?|2|2|
|4|How many toys does he have now?|5 + 2 + 2|9|
Therefore, the answer (arabic numerals) is 9.

There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many computers were in the server room?|9|9|
|2|How many computers were installed each day?|5|5|
|3|How many computers are now in the server room?|9 + 5 + 5 + 5 + 5|29|
Therefore, the answer (arabic numerals) is 29.

Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

|step|subquestion|process|result|
|---|---|---|---|
|1|How many golf balls did Michael have?|58|58|
|2|How many golf balls did he lose on tuesday?|23|23|
|3|How many golf balls did he lose on wednesday?|2|2|
|4|How many golf balls did he have at the end of wednesday?|58 - 23 - 2|33|
Therefore, the answer (arabic numerals) is 33.

Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

|step|subquestion|process|result|
|---|---|---|---|
|1|How much money does Olivia have?|23|23|
|2|How much does each bagel cost?|3|3|
|3|How many bagels did she buy?|5|5|
|4|How much money did she spend on bagels?|3 x 5|15|
|5|How much money does she have left?|23 - 15|8|
Therefore, the answer (arabic numerals) is 8.

'''
    
    else:
        raise ValueError("dataset is not properly defined ...")
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text
