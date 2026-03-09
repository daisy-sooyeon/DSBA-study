import logging
import wandb
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict
from accelerate import Accelerator
from utils.utils import check_graph, Float32Encoder
from utils.tools import adjust_learning_rate, EarlyStopping
from utils.metrics import MSE, cal_metric, anomaly_metric, bf_search, calc_seq, get_best_f1, get_adjusted_composite_metrics, percentile_search, bf_search1, calc_seq1, pot_eval, safe_auc, hit_att, ndcg

_logger = logging.getLogger('train')

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training_dl(
    model, trn_dataloader, val_dataloader, criterion, optimizer, accelerator: Accelerator, 
    savedir: str, epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, 
    use_wandb: bool, wandb_iter: int, ckp_metric: str, model_name: str, 
    early_stopping_metric: str, early_stopping_count: int,
    lradj: int, learning_rate: int, model_config: dict):
    
    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # set mode
    model.train()
    optimizer.zero_grad()
    end_time = time.time()
    
    early_stopping = EarlyStopping(patience=early_stopping_count)
    mse_criterion = nn.MSELoss()
    
    # init best score and step
    best_score = np.inf
    wandb_iteration = 0
    
    _logger.info(f"\n 🔹 Training started")

    for epoch in range(epochs):
        epoch_time = time.time()
        for idx, item in enumerate(trn_dataloader):
            data_time_m.update(time.time() - end_time)

            """
            목적: 구성한 Dataloader를 바탕으로 모델의 입력을 구성
            조건
            - 구성한 Dataloader에 적합한 입력을 통하여 모델의 출력을 계산
            - 이상탐지 모델은 loss나 score를 계산하는 과정이 모델마다 상이할 수 있기에 모델 내부에서 계산
            - model은 LSTM_AE를 사용하고 있기 때문에, 코드 참고하여 작성
            - 모든 모델에서 모델만 변경할 경우 작동될 수 있도록 구현
            """

            batch_x = item[0].float().to(accelerator.device)
            # 트랜스포머 입력에 맞게 [Seq, Batch, Feat]로 차원 변경
            src = batch_x.transpose(0, 1) 
            tgt = batch_x.transpose(0, 1)

            x1, x2 = model(src, tgt)

            weight = 1.0 / (epoch + 1)
            l1 = mse_criterion(x1, tgt)
            l2 = mse_criterion(x2, tgt)
            loss = (weight * l1) + ((1 - weight) * l2)

            loss = accelerator.gather(loss).mean()
            outputs = x2.transpose(0, 1) # [Batch, Seq, Feat]
            outputs, targets = accelerator.gather_for_metrics((outputs.contiguous(), tgt.contiguous()))

            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            
            losses_m.update(loss.item(), n = targets.size(0))
            
            # batch time
            batch_time_m.update(time.time() - end_time)
            wandb_iteration += 1
            
            if use_wandb and (wandb_iteration+1) % wandb_iter:
                train_results = OrderedDict([
                    ('lr',optimizer.param_groups[0]['lr']),
                    ('train_loss',losses_m.avg)
                ])
                wandb.log(train_results, step=wandb_iteration)
        
        # meta learning
        meta_batch = next(iter(trn_dataloader))
        meta_x = meta_batch[0].float().to(accelerator.device)
        meta_src, meta_tgt = meta_x.transpose(0, 1), meta_x.transpose(0, 1)

        current_lr = optimizer.param_groups[0]['lr']
        decay_ratio = current_lr / learning_rate 
        
        decayed_meta_lr = 0.02 * decay_ratio
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = decayed_meta_lr

        meta_x1, meta_x2 = model(meta_src, meta_tgt)
        weight = 1.0 / (epoch + 1)
        meta_l1 = mse_criterion(meta_x1, meta_tgt)
        meta_l2 = mse_criterion(meta_x2, meta_tgt)
        meta_loss = (weight * meta_l1) + ((1 - weight) * meta_l2)
        meta_loss = accelerator.gather(meta_loss).mean()

        accelerator.backward(meta_loss)
        optimizer.step()
        optimizer.zero_grad()

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        _logger.info(f"   ✨ [Meta-Learning] Applied MAML on a random batch (Meta Loss: {meta_loss.item():.4f})")

        if (epoch+1) % log_epochs == 0:
            _logger.info('EPOCH {:>3d}/{} | TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (epoch+1), epochs, 
                        (idx+1), 
                        len(trn_dataloader), 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = batch_x.size(0) / batch_time_m.val,
                        rate_avg   = batch_x.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
            
                    
        if (epoch+1) % eval_epochs == 0:
            eval_metrics = test_dl(
                accelerator   = accelerator,
                model         = model, 
                dataloader    = val_dataloader, 
                criterion     = criterion,
                name          = 'VALID',
                log_interval  = log_eval_iter,
                savedir       = savedir,
                model_name    = model_name,
                model_config  = model_config,
                return_output = False,
                trn_dataloader=trn_dataloader
                )

            model.train()
            
            # eval results
            eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
            
            # wandb
            if use_wandb:
                wandb.log(eval_results, step=wandb_iteration)
                
            # check_point
            if best_score > eval_metrics[ckp_metric]:
                # save results
                state = {'best_epoch':epoch ,
                            'best_step':idx+1, 
                            f'best_{ckp_metric}':eval_metrics[ckp_metric]}
                
                print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                    to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    state.update(eval_results)
                    json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), 
                                indent='\t', cls=Float32Encoder)

                # save model
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))
                    _logger.info("\n✅ Saved best model")
                best_score = eval_metrics[ckp_metric]
                
            early_stopping(eval_metrics[early_stopping_metric])
            if early_stopping.early_stop:
                _logger.info("⏳ Early stopping triggered")
                break
        
        adjust_learning_rate(optimizer, epoch + 1, lradj, learning_rate)

        end_time = time.time()

    # save latest model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

        print('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

        # save latest results
        state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
        state.update(eval_results)
        json.dump(state, open(os.path.join(savedir, f'latest_results.json'),'w'), indent='\t', cls=Float32Encoder)
    _logger.info("\n🎉 Training complete for all datasets")
    
def test_dl(model, dataloader, criterion, accelerator: Accelerator, log_interval: int, 
            savedir: str, model_config: dict, model_name: str, name: str = 'TEST', 
            return_output: bool = False, plot_result:bool = False, trn_dataloader = None) -> dict:
    _logger.info(f'\n[🔍 Start {name} Evaluation]')

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    total_label = []
    total_outputs = []
    total_score   = []
    total_targets = []
    total_timestamp = []
    history = dict()

    mse_criterion = nn.MSELoss()

    end_time = time.time()

    model.eval()

    train_scores_np = None
    if name == 'TEST' and trn_dataloader is not None:
        _logger.info("📊 Calculating training scores for POT thresholding...")
        train_scores = []
        with torch.no_grad():
            for item in trn_dataloader:
                batch_x = item[0].float().to(accelerator.device)
                src = batch_x.transpose(0, 1)
                tgt = batch_x.transpose(0, 1)

                x1, x2 = model(src, tgt)
                x1_out = x1.transpose(0, 1)
                x2_out = x2.transpose(0, 1)
                score = 0.5 * torch.mean((x1_out - batch_x)**2, dim=-1) + 0.5 * torch.mean((x2_out - batch_x)**2, dim=-1)
                train_scores.append(score.detach().cpu().numpy())

        train_scores_np = np.concatenate(train_scores, axis=0)[:, -1]
        
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            data_time_m.update(time.time() - end_time)

            """
            목적: 구성한 Dataloader를 바탕으로 모델의 입력을 구성
            조건
            - 구성한 Dataloader에 적합한 입력을 통하여 모델의 출력을 계산
            - model은 LSTM_AE를 사용하고 있기 때문에, 코드 참고하여 작성
            """

            batch_x = item[0].float().to(accelerator.device)
            batch_y = item[1].float().to(accelerator.device) if len(item) > 1 else None

            src = batch_x.transpose(0, 1)
            tgt = batch_x.transpose(0, 1)

            x1, x2 = model(src, tgt)

            loss = 0.5 * mse_criterion(x1, tgt) + 0.5 * mse_criterion(x2, tgt)
            loss = accelerator.gather(loss).mean()
            losses_m.update(loss.item(), n=batch_x.size(0))

            x1_out = x1.transpose(0, 1)
            x2_out = x2.transpose(0, 1)
            score_1 = torch.mean((x1_out - batch_x)**2, dim=-1)
            score_2 = torch.mean((x2_out - batch_x)**2, dim=-1)
            score = 0.5 * score_1 + 0.5 * score_2

            outputs = x2_out.detach().cpu().numpy()
            targets = tgt.detach().cpu().numpy()
            score = score.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_score.append(score)
            total_targets.append(targets)
            # total_timestamp.append(input_timestamp.detach().cpu().numpy())

            if name == 'TEST' and len(item) > 1:
                label = item[1].detach().cpu().numpy()
                total_label.append(label)

            batch_time_m.update(time.time() - end_time)

            if (idx+1) % log_interval == 0:
                _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (idx+1), 
                                len(dataloader),
                                name       = name, 
                                loss       = losses_m, 
                                batch_time = batch_time_m,
                                rate       = batch_x.size(0) / batch_time_m.val,
                                rate_avg   = batch_x.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))

            end_time = time.time()

    
    """
    목적: 시계열 이상탐지 Task의 평가 지표 계산
    조건
    - 계산된 출력, 입력, label, score 등을 가지고, 시계열 이상탐지 metric 계산
    - 'metrics.py'의 cal_metric, bf_search, calc_seq 함수 참고하여 작성
    - 'VALID' 시에는 reconstruction loss만 도출
    """
    if name == 'VALID':
        history['MSE'] = losses_m.avg
        # pred_scores = np.concatenate(total_score, axis=0)[:, -1]
        
        # true_labels = np.concatenate(total_label, axis=0)[:, -1]
        # results = cal_metric(true_labels, pred_scores)
        
        # if isinstance(results, dict):
        #     history.update({f'val_{k}': v for k, v in results.items()})

        _logger.info(f"✨ VALID RESULT | MSE Loss: {losses_m.avg:.4f}")

    elif name == 'TEST':
        # [Total_Samples, Seq_len] -> 윈도우 마지막 시점 추출
        pred_scores = np.concatenate(total_score, axis=0)[:, -1]
        true_labels = np.concatenate(total_label, axis=0)[:, -1]

        _logger.info("🔍 [POT] Running Peaks-Over-Threshold Algorithm...")
        
        # POT 알고리즘으로 임계값 및 기본 지표 산출
        pot_res, preds = pot_eval(train_scores_np, pred_scores, true_labels, model_config.pot_q, model_config.pot_level)
        
        auc_score = safe_auc(true_labels, pred_scores)
        pot_res['AUC'] = auc_score
        
        hitrate_res_1 = hit_att(pred_scores, true_labels)
        ndcg_res_1 = ndcg(pred_scores, true_labels)

        hitrate_res_15 = hit_att(pred_scores, true_labels, 1.5)
        ndcg_res_15 = ndcg(pred_scores, true_labels, 1.5)
        
        pot_res.update(hitrate_res_1)
        pot_res.update(ndcg_res_1)

        pot_res.update(hitrate_res_15)
        pot_res.update(ndcg_res_15)

        history.update(pot_res)
        history['test_loss'] = losses_m.avg

        _logger.info(f"🏆 {name} RESULT | F1: {pot_res['f1']:.4f} | AUC: {auc_score:.4f} | Prec: {pot_res['precision']:.4f} | Rec: {pot_res['recall']:.4f}")
        _logger.info(f"🏆 {name} RANKING | {hitrate_res_15} | {ndcg_res_15} | {hitrate_res_1} | {ndcg_res_1}")

        if accelerator.is_main_process:
            try:
                plt.figure(figsize=(15, 5))
                
                # 1) Anomaly Score (주황색 선)
                plt.plot(pred_scores, label='Anomaly Score', color='darkorange', linewidth=1.5)
                
                # 2) POT Threshold (빨간색 점선)
                plt.axhline(y=pot_res['threshold'], color='red', linestyle='--', label=f"POT Threshold ({pot_res['threshold']:.4f})")
                
                # 3) 실제 이상치 구간 (Ground Truth) 하이라이트 표시
                anom_indices = np.where(true_labels == 1)[0]
                plt.scatter(anom_indices, pred_scores[anom_indices], color='red', label='True Anomaly (Label=1)', s=15, zorder=3)
                
                # 4) 예측된 이상치 영역(Predicted) 색칠
                if 'preds' in locals() or 'preds' in globals():
                    pred_indices = np.where(preds == 1)[0]
                    if pred_indices.size > 0:
                        # 연속된 구간으로 묶어서 배경에 연하게 칠해준다
                        segments = np.split(pred_indices, np.where(np.diff(pred_indices) != 1)[0] + 1)
                        for seg in segments:
                            start, end = seg[0], seg[-1]
                            plt.axvspan(start, end + 1, color='blue', alpha=0.15)
                        plt.scatter(pred_indices, pred_scores[pred_indices], color='blue', label='Predicted Anomaly', s=10, zorder=2)
                
                plt.title(f'Anomaly Score & Threshold ({name})', fontsize=16, fontweight='bold')
                plt.xlabel('Time Steps', fontsize=12)
                plt.ylabel('Score (MSE)', fontsize=12)
                
                # 범례를 우측 상단에 배치
                plt.legend(fontsize=12, loc='upper right')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                
                save_path = os.path.join(savedir, f'{name}_anomaly_plot.png')
                plt.savefig(save_path, dpi=300)
                plt.close()
                
                _logger.info(f"📊 {name} Anomaly plot saved to: {save_path}")
            except Exception as e:
                _logger.warning(f"⚠️ Failed to draw prediction plot: {e}")

    return history