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

def load_task(task_name: str, evaluation_api: EngineLM,  *args, **kwargs) :

    if "binary_classification" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from textgrad.tasks.binary_classification import CLS_binary
        
        task_name = "hyoje/cls_binary"
        print('task')
        train_set = CLS_binary(task_name, split="train")
        print('train_cls_finish')
        val_set = CLS_binary(task_name, split="valid")
        print('valid_cls_finish')
        test_set = CLS_binary(task_name, split="test")
        
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]

        evaluation_instruction = "Below is a prompt and examples from text-generation task, the ground truth answer. If the final generated note are note easily distinguishable from the example samples, say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        
        return train_set, val_set, test_set, eval_fn
    
        
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
