import numpy as np
import torch
import torch.utils.data
import pickle
import constants 
from math import ceil


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, raw_data, interval_scale=None):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        data = []
        for inst in raw_data:
            for i in range(ceil(len(inst) / constants.MAX_LEN)):
                indices = slice(i * constants.MAX_LEN, (i + 1) * constants.MAX_LEN)
                if len(inst[indices]) < 2:
                    continue
                data.append(inst[indices])
        
        # scaling timestamp values to normal range
        if interval_scale is None:
            self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
            self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        else:
            self.time = [[elem['time_since_start'] / interval_scale for elem in inst] for inst in data]
            self.time_gap = [[elem['time_since_last_event'] / interval_scale for elem in inst] for inst in data]
        
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        
        self.comp = []
        for inst in data:
            inst_comps = []
            for elem in inst:
                if 'comp' in elem.keys():
                    c = elem['comp'] + 1
                else:
                    c = 0
                inst_comps.append(c)
            self.comp.append(inst_comps)
                               

        self.length = len(self.time)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.comp[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    insts.sort(key=lambda x: len(x[0]), reverse=True)
    
    time, time_gap, event_type, comp = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    comp = pad_type(comp)
    return time, time_gap, event_type, comp


def get_dataloader(data, batch_size, interval_scale=None, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data, interval_scale)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl


def prepare_dataloader(opt, interval_scale=None):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, interval_scale, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, interval_scale, shuffle=False)
    return trainloader, testloader, num_types
