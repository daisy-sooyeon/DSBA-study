from torch.utils.data import Dataset
import numpy as np

class BuildDataset(Dataset):
    def __init__(self, data: np.ndarray, data_stamp: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.data_stamp = data_stamp
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):        
        # 슬라이딩 윈도우로 만들 수 있는 총 데이터(조각)의 개수
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # input의 시작과 끝 인덱스
        s_begin = idx
        s_end = s_begin + self.seq_len

        # output의 시작과 끝 인덱스
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark