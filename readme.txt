Setup the environment from requirements.txt: 
pip install -r requirements.txt

To use our code, you must have your own OpenAI API from: https://beta.openai.com/account/api-keys, and put it in util.py:62

Here is an example script to run our method on SingleEq dataset under Zero-shot setting:
python main.py --model=gpt3-xl --dataset=singleeq --limit_dataset_size 0 --model=code-xl --method=zero_shot_cot

Here is an example script to run our method on SingleEq dataset under Zero-shot setting with self-consistency:
python main.py --model=gpt3-xl --dataset=singleeq --limit_dataset_size 0 --model=code-xl --method=zero_shot_cot --self_consistency

Here is an example script to run our method on SingleEq dataset under few-shot setting with self-consistency:
python main.py --model=gpt3-xl --dataset=singleeq --limit_dataset_size 0 --model=code-xl --method=few_shot_cot