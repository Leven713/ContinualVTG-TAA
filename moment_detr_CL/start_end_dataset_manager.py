from moment_detr_CL.start_end_dataset_all import StartEndDataset
from typing import Optional
import numpy as np


class DataManager(object):
    def __init__(self, dataset: StartEndDataset, task_num: int, train: bool = True, shuffle_seed: Optional[int] = None, sequence_seed: Optional[int] = None):
        """
        Args:
            dataset (StartEndDataset): The dataset to manage.
            task_num (int): The number of tasks.
            train (bool): Whether the dataset is for training or not.
            permute_seed (Optional[int]): Seed for shuffling the dataset.
            sequence_seed (Optional[int]): Seed for sequence generation.
        """
        self.dataset = dataset
        self.task_num = task_num
        self.train = train
        self.datalist = dataset.data
        self.task_size = len(self.datalist)//task_num

        # dataset permutation
        if shuffle_seed != -1:
            rng_state = np.random.get_state()
            np.random.seed(shuffle_seed)
            np.random.shuffle(self.datalist)    
            np.random.set_state(rng_state)

        # task sequence
        if sequence_seed != -1:    
            rng_state = np.random.get_state()
            np.random.seed(sequence_seed)
            self.learning_sequence = np.random.permutation(self.task_num)
            np.random.set_state(rng_state)
        else:
            self.learning_sequence = np.arange(self.task_num)

        print('learning_sequence:', self.learning_sequence)



    def get_dataset(self, task_id: int):
        assert 0 <= task_id < self.task_num, "Task ID out of range"

        real_task_id = self.learning_sequence[task_id]   # cur-task ID
        start_index = real_task_id * self.task_size
        end_index = start_index + self.task_size
        if end_index > len(self.datalist):
            end_index = len(self.datalist)
        task_data = self.datalist[start_index:end_index]

        self.dataset.data = task_data
        return self.dataset
