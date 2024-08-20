import platformdirs

from .base import Dataset

class CLS(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="train", *args, **kwargs):
        """
        GSM8K dataset from HF."""
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "validation", "test"]
        if split == "test":
            self.data = load_dataset("hyoje/cls_binary", subset, cache_dir=root, split="test")
        elif split == "validation":
            self.data = load_dataset("hyoje/cls_binary", subset, cache_dir=root, split="validation")
        elif split == "train":
            self.data = load_dataset("hyoje/cls_binary", subset, cache_dir=root, split="train")
        self.split = split
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        answer = row["answer"]
        question_prompt = f"Question: {question}"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will generate the progress notes and discharge note referring to the several examples. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."


class CLS_binary(CLS):
    def __init__(self, root:str=None, split: str="train"):
        import tqdm
        import random
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        dataset = load_dataset("hyoje/cls_binary", cache_dir=root)
        hf_official_train = dataset['train'][:5]
        hf_official_valid = dataset['validation'][:2]
        hf_official_test = dataset['test'][:2]
        official_train = []
        official_valid = []
        official_test = []
        print("******************")
        for example in tqdm.tqdm(hf_official_train):
            question = example['question']
            answer = example['answer']
            official_train.append(dict(question=question, answer=answer))
        print("//////////////////////////////")
        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer']
            official_test.append(dict(question=question, answer=answer))
        print("/////**************//") 
        for example in tqdm.tqdm(hf_official_valid):
            question = example['question']
            answer = example['answer']
            official_valid.append(dict(question=question, answer=answer))
          
        rng = random.Random(0)
        rng.shuffle(official_train)
        rng = random.Random(0)
        rng.shuffle(official_valid)
        rng = random.Random(0)
        rng.shuffle(official_test)
        trainset = official_train[:]
        devset = official_valid[:]
        testset = official_test[:]
        if split == "train":
            self.data = trainset
        elif split == "validation":
            self.data = devset
        elif split == "test":
            self.data = testset
