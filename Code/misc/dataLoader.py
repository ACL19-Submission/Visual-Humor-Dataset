import json
import h5py
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, input_text_file, input_video_file, input_json, data_split):
        self.f_text = h5py.File(input_text_file, "r")
        f = json.load(open(input_json, 'r'))
        self.idx2d_id = f['itod' + data_split.capitalize()]
        self.itow_dialog = f['vocab_i2wDialog']  # vocabulary file for the Dialog
        self.data_split = data_split
        self.vocab_size = len(self.itow_dialog) + 1

    def __len__(self):
        return (len(self.idx2d_id))

    def __getitem__(self, idx):
        d_id = self.idx2d_id[str(idx)]
        dialogTurns = np.array(self.f_text[(self.data_split).lower()][str(d_id)]["DialogTurns"])
        label = np.array((self.f_text)[(self.data_split).lower()][str(d_id)]["DialogGT"])
        dialogLength = dialogTurns.shape[0]
        turnLengths = np.zeros(dialogLength)
        for it in range(dialogLength):
            turnLengths[it] = int((np.where(dialogTurns[it] == self.END_TOKEN))[0] + 1)

        return {"dialog_turns": dialogTurns, "label": label, "dialog_id": d_id, "turn_lengths": turnLengths}
