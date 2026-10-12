import random
import torch
from utils.tensor_utils import pad_sequences_1d


class ReservoirBuffer:
    """
    Reservoir buffer for experience replay.
    """

    def __init__(self, buffer_size_per_task):
        # buffer to store (x, y) pairs
        self.n_seen_examples = 0
        self.x_buffer = None
        self.y_buffer = None
        self.task_id_buffer = None
        self.is_filled = None
        self.buffer_size_per_task = buffer_size_per_task

    
    def reset_new_buffer(self):
        # reset the new buffer for the current task
        if self.x_buffer is not None and len(self.x_buffer) > 0:
            self.x_buffer_new = {}
            for k, v in self.x_buffer.items():
                if k in ["src_txt", "src_vid"]:
                    self.x_buffer_new[k] = [None] * self.buffer_size_per_task
            
            self.y_buffer_new = {}
            for k, v in self.y_buffer.items():
                if isinstance(v, torch.Tensor):
                    self.y_buffer_new[k] = torch.empty(
                        (self.buffer_size_per_task, *v.shape[1:]), 
                        dtype=v.dtype, 
                        device=v.device
                    )
                else:
                    self.y_buffer_new[k] = [None] * self.buffer_size_per_task
            
            if self.task_id_buffer is not None:
                self.task_id_buffer_new = torch.empty(
                    (self.buffer_size_per_task, *self.task_id_buffer.shape[1:]), 
                    dtype=self.task_id_buffer.dtype, 
                    device=self.task_id_buffer.device
                )
            else:
                self.task_id_buffer_new = torch.empty(
                    self.buffer_size_per_task, 
                    dtype=torch.long, 
                    device=self.device
                )
            
            if self.is_filled is not None:
                self.is_filled_new = torch.empty(
                    (self.buffer_size_per_task, *self.is_filled.shape[1:]), 
                    dtype=self.is_filled.dtype, 
                    device=self.is_filled.device
                )
            else:
                self.is_filled_new = torch.empty(
                    self.buffer_size_per_task, 
                    dtype=torch.bool, 
                    device=self.device
                )
        else:
            # if no existing buffer, initialize new buffer as None
            self.x_buffer_new = None
            self.y_buffer_new = None
            self.task_id_buffer_new = None
            self.is_filled_new = None

        print('Reset New Buffer!')



    def merge_buffers(self):
        # merge new buffer into the total buffer
        if self.x_buffer is None:
            self.x_buffer = self.x_buffer_new
            self.y_buffer = self.y_buffer_new
            self.task_id_buffer = self.task_id_buffer_new
            self.is_filled = self.is_filled_new
        else:
            for k, v in self.x_buffer_new.items():
                self.x_buffer[k].extend(v)
            for k, v in self.y_buffer_new.items():
                if isinstance(v, torch.Tensor):
                    self.y_buffer[k] = torch.cat((self.y_buffer[k], v), dim=0)
                else:
                    self.y_buffer[k].extend(v)
            self.task_id_buffer = torch.cat((self.task_id_buffer, self.task_id_buffer_new), dim=0)
            self.is_filled = torch.cat((self.is_filled, self.is_filled_new), dim=0)

        print('Merge New Buffer!')
        print('len(buffer): ', self.is_filled.shape[0])



    def add(self, x, y, task_id=None):
        """Adds data (x, y) to the buffer using reservoir sampling."""
        ## initialize new buffer if not existing
        if self.x_buffer_new is None:
            self.x_buffer_new = {}
            for k, v in x.items():
                # print(k, v.shape)
                if k in ["src_txt", "src_vid"]:
                    self.x_buffer_new[k] = [None]*self.buffer_size_per_task

            self.y_buffer_new = {k: torch.empty((self.buffer_size_per_task, *v.shape[1:]), dtype=v.dtype, device=v.device) 
                                if isinstance(v, torch.Tensor)    
                                else [None]*self.buffer_size_per_task   
                                    for k, v in y.items()}
            
            self.task_id_buffer_new = torch.empty((self.buffer_size_per_task, 1), dtype=torch.int64)
            self.is_filled_new = torch.zeros(self.buffer_size_per_task, dtype=torch.bool)

        for i in range(len(next(iter(x.values())))):   
            sample_x = {}
            for k, v in x.items():
                if k in ["src_txt", "src_vid"]:
                    sample_x[k] = v[i][x[f"{k}_mask"][i].bool()]
            sample_y = {k: v[i] for k, v in y.items()}

            self._add_single(sample_x, sample_y, task_id=task_id)


    def add_y_buffer(self, k, v, index):
        if isinstance(v, torch.Tensor):
            try:
                self.y_buffer_new[k][index] = v.clone().detach()
            except RuntimeError as e:
                if "expanded size" in str(e) or "must match" in str(e):
                    buffer_shape = self.y_buffer_new[k].shape[1:] 
                    v_shape = v.shape
                    new_tensor = torch.zeros(buffer_shape, dtype=v.dtype, device=v.device)
                    if v.dim() == len(buffer_shape):
                        min_dims = [min(s1, s2) for s1, s2 in zip(buffer_shape, v_shape)]
                        if all(m > 0 for m in min_dims): 
                            src_slices = tuple(slice(0, dim) for dim in min_dims)
                            dst_slices = tuple(slice(0, dim) for dim in min_dims)
                            new_tensor[dst_slices] = v[src_slices]
                    
                    self.y_buffer_new[k][index] = new_tensor
                else:
                    raise e
        else:
            self.y_buffer_new[k][index] =  {k_: v_.clone().detach() for k_, v_ in v.items()}  


    def _add_single(self, x, y, task_id=None):
        """Helper function to add a single data point."""
        self.n_seen_examples += 1
        if not self.is_filled_new.all():
            index = self.is_filled_new.logical_not().nonzero(as_tuple=True)[0][0]

            for k, v in x.items():
                self.x_buffer_new[k][index] = v.clone().detach()
                
            for k, v in y.items():
                self.add_y_buffer(k, v, index)

            if task_id is not None:
                self.task_id_buffer_new[index] = task_id
            self.is_filled_new[index] = True
        else:
            # reservoir sampling
            replace_index = random.randint(0, self.n_seen_examples + 1)
            if replace_index < self.buffer_size_per_task:
                for k, v in x.items():
                    self.x_buffer_new[k][replace_index] = v.clone().detach()
                for k, v in y.items():
                    self.add_y_buffer(k, v, replace_index)
                if task_id is not None:
                    self.task_id_buffer_new[replace_index] = task_id

    def sample(self, batch_size=1):
        """Returns batch_size random samples (x, y) from the buffer."""
        if self.x_buffer is None:
            return None

        valid_indices = torch.where(self.is_filled)[0]  
        
        non_none_indices = []
        for idx in valid_indices:
            is_valid = True
            for k, v in self.x_buffer.items():
                if k in ["src_txt", "src_vid"] and v[idx] is None:
                    is_valid = False
                    break
            if is_valid:
                non_none_indices.append(idx)
        
        if len(non_none_indices) == 0:
            return None
        
        non_none_indices = torch.tensor(non_none_indices, device=valid_indices.device)
        batch_size = min(batch_size, len(non_none_indices)) 
        indices = torch.randint(0, len(non_none_indices), (batch_size,))   

        task_ids = None
        if self.task_id_buffer is not None:
            task_ids = self.task_id_buffer[indices]

        real_indices = non_none_indices[indices] 

        sampled_x = {}
        for k, v in self.x_buffer.items():
            if k in ["src_txt", "src_vid"]:
                try:
                    # for test
                    sequences = [v[idx] for idx in real_indices]
                    valid_sequences = [seq for seq in sequences if seq is not None]
                    if len(valid_sequences) == 0:
                        continue
                        
                    pad_data = pad_sequences_1d(valid_sequences, dtype=torch.float32, 
                                            device=valid_sequences[0].device if len(valid_sequences) > 0 else torch.device('cpu'), 
                                            fixed_length=None)
                    
                    if k == "src_txt":
                        sampled_x["src_txt"] = pad_data[0]
                        sampled_x["src_txt_mask"] = pad_data[1]
                    else:
                        sampled_x["src_vid"] = pad_data[0]
                        sampled_x["src_vid_mask"] = pad_data[1]
                        
                except Exception as e:
                    print(f"Error processing {k}: {e}")
                    print(f"real_indices: {real_indices}")
                    print(f"sequences lengths: {[len(seq) if seq is not None else 'None' for seq in sequences]}")
                    raise e

        sampled_y = {}
        for k, v in self.y_buffer.items():
            if isinstance(v, torch.Tensor):
                sampled_y[k] = v[real_indices]
            else:
                sampled_y[k] = [v[idx] for idx in real_indices]

        return sampled_x, sampled_y, task_ids


    def __len__(self):
        return self.n_seen_examples

    def is_empty(self):
        return self.x_buffer is None
