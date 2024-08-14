from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader
from .leetcode import LeetCodeHardEval

from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM



AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_word_sorting",
    "GSM8K_DSPy",
    "binary_classification"
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond"
    "LeetCodeHardEval"
]

def load_task(df_train: str, df_valid: str, df_test: str, task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:

    if task_name = "binary_classification" :
        
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard

        train_data = pd.read_csv(df_train)
        valid_data = pd.read_csv(df_valid)
        test_data = pd.read_csv(df_test)

        
        evaluation_instruction = "Below is a prompt from text-generation task. Is the final result similar, i.e. the similar to the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        return train_set, val_set, test_set, eval_fn

    
    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        
        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        return train_set, val_set, test_set, eval_fn
    
            
    elif task_name = "binary_classification" :
        
    else:
        raise ValueError(f"Task {task_name} not found.")


def load_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if "MMLU_" in task_name:
        subset = task_name[5:]
        test_set = MMLUInstanceDataset(evaluation_api=evaluation_api, subset=subset, split="test", *args, **kwargs)
        return test_set
    elif "GPQA" in task_name:
        from .gpqa import GPQAInstanceDataset
        test_set = GPQAInstanceDataset(evaluation_api=evaluation_api, subset=task_name.lower(), *args, **kwargs)
        return test_set
    elif task_name in ["LeetCodeHardEval"]:
        dataset = LeetCodeHardEval()
        return dataset
    else:
        raise ValueError(f"Instance task {task_name} not found.")
