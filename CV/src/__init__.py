"""
CV 모델 학습 및 평가 스크립트 모음
"""

def train_main(cfg):
    """모델 학습"""
    from .train import main
    return main(cfg)


def finetune_main(cfg, weights_path=None):
    """모델 미세조정"""
    from .finetune import main
    return main(cfg, weights_path)


def eval_robustness_main(cfg, weights_path):
    """강건성 평가"""
    from .eval_robustness import main
    return main(cfg, weights_path)


def eval_background_robustness_main(cfg):
    """배경 편향 강건성 평가"""
    from .eval_background_robustness import main
    return main(cfg)


def eval_background_robustness_main(cfg):
    """배경 편향 강건성 평가"""
    from .eval_background_robustness import main
    return main(cfg)


__all__ = [
    'train_main',
    'finetune_main',
    'eval_robustness_main',
    'eval_background_robustness_main',
]
