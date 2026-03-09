import numpy as np
import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, data, data_stamp, seq_len: int, stride_len: int = 1, labels=None):
        """
        TranAD 및 이상 탐지를 위한 슬라이딩 윈도우(Sliding Window) 데이터셋
        
        Args:
            data (np.ndarray): (전체 시점 수, 변수 개수) 형태의 스케일링된 센서 데이터
            data_stamp (np.ndarray): (전체 시점 수, 1) 형태의 시간 스탬프 데이터
            seq_len (int): 한 번에 모델에 넣을 윈도우 크기 (과거 데이터 길이)
            stride_len (int): 윈도우를 몇 칸씩 이동시키면서 자를 것인지 (보통 1)
            labels (np.ndarray, optional): 이상치 정답지 (Test 셋에만 존재)
        """
        self.data = data
        self.data_stamp = data_stamp
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.labels = labels

        # 생성 가능한 총 윈도우(조각)의 개수
        self.num_samples = (len(self.data) - self.seq_len) // self.stride_len + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> tuple:
        # 윈도우의 시작과 끝 인덱스 계산
        start_idx = idx * self.stride_len
        end_idx = start_idx + self.seq_len

        window_data = self.data[start_idx:end_idx]

        # target 정답 라벨
        if self.labels is not None: # Test 셋인 경우
            window_target = self.labels[start_idx:end_idx]
        else:
            # Train/Valid 셋인 경우
            window_target = window_data 

        return window_data, window_target, np.zeros(1), np.zeros(1)