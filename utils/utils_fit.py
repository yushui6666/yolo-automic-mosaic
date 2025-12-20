#-----------------------------------------------------------------------#
#   utils_fit.py：训练一个 epoch 的核心函数
#   功能概述：
#   1. 执行一个完整的训练 epoch（训练 + 验证）
#   2. 支持混合精度训练（FP16）
#   3. 支持 EMA（指数移动平均）模型更新
#   4. 自动保存最佳模型和定期检查点
#-----------------------------------------------------------------------#
import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    训练一个完整的 epoch
    
    参数:
        model_train: torch.nn.Module, 训练模式下的模型
        model: torch.nn.Module, 原始模型（用于保存）
        ema: EMA 对象或 None, 指数移动平均模型（用于稳定训练）
        yolo_loss: 损失函数对象
        loss_history: LossHistory 对象, 用于记录损失
        eval_callback: EvalCallback 对象, 用于评估模型
        optimizer: torch.optim.Optimizer, 优化器
        epoch: int, 当前 epoch 编号
        epoch_step: int, 每个 epoch 的训练步数
        epoch_step_val: int, 每个 epoch 的验证步数
        gen: DataLoader, 训练数据加载器
        gen_val: DataLoader, 验证数据加载器
        Epoch: int, 总 epoch 数
        cuda: bool, 是否使用 GPU
        fp16: bool, 是否使用混合精度训练
        scaler: torch.cuda.amp.GradScaler 或 None, FP16 训练的梯度缩放器
        save_period: int, 模型保存周期（每 save_period 个 epoch 保存一次）
        save_dir: str, 模型保存目录
        local_rank: int, 分布式训练的本地 GPU 索引，默认 0
    
    功能:
        1. 训练阶段：前向传播、计算损失、反向传播、更新参数
        2. 验证阶段：评估模型性能
        3. 记录损失和评估指标
        4. 保存模型检查点（定期保存、最佳模型、最新模型）
    """
    # 初始化损失累加器
    loss        = 0  # 训练损失累加
    val_loss    = 0  # 验证损失累加

    # ==================== 训练阶段 ====================
    if local_rank == 0:
        print('Start Train')
        # 创建训练进度条
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    # 设置模型为训练模式（启用 dropout、batch normalization 的训练行为）
    model_train.train()
    
    # 遍历训练数据
    for iteration, batch in enumerate(gen):
        # 防止超出指定的训练步数
        if iteration >= epoch_step:
            break

        # 取出图像和标签
        images, bboxes = batch
        
        # 将数据移到 GPU（不计算梯度，因为只是数据传输）
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
        
        #----------------------#
        #   清零梯度（重要：每次迭代前必须清零）
        #----------------------#
        optimizer.zero_grad()
        
        # 根据是否使用混合精度训练，选择不同的训练流程
        if not fp16:
            # ========== FP32 训练流程 ==========
            #----------------------#
            #   前向传播：模型推理
            #   输出包含：dbox, cls, origin_cls, anchors, strides
            #----------------------#
            outputs = model_train(images)
            # 计算损失
            loss_value = yolo_loss(outputs, bboxes)
            
            #----------------------#
            #   反向传播：计算梯度
            #----------------------#
            loss_value.backward()
            
            # 梯度裁剪：防止梯度爆炸（将梯度范数限制在 10.0 以内）
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            
            # 更新模型参数
            optimizer.step()
        else:
            # ========== FP16 混合精度训练流程 ==========
            from torch.cuda.amp import autocast
            
            # 在 autocast 上下文中进行前向传播（自动使用 FP16）
            with autocast():
                #----------------------#
                #   前向传播（FP16）
                #----------------------#
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            #----------------------#
            #   反向传播（FP16，带梯度缩放）
            #----------------------#
            # 缩放损失（防止 FP16 下梯度下溢）
            scaler.scale(loss_value).backward()
            # 取消缩放梯度（用于梯度裁剪）
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            # 更新参数（scaler 会自动处理缩放）
            scaler.step(optimizer)
            # 更新缩放因子
            scaler.update()
        
        # 更新 EMA 模型（如果启用）
        if ema:
            ema.update(model_train)

        # 累加损失
        loss += loss_value.item()
        
        # 更新进度条（只在主进程显示）
        if local_rank == 0:
            pbar.set_postfix(**{
                'loss': loss / (iteration + 1),  # 平均损失
                'lr': get_lr(optimizer)         # 当前学习率
            })
            pbar.update(1)

    # ==================== 验证阶段 ====================
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        # 创建验证进度条
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # 选择用于验证的模型（优先使用 EMA 模型，更稳定）
    if ema:
        model_train_eval = ema.ema  # 使用 EMA 模型
    else:
        model_train_eval = model_train.eval()  # 使用普通模型，设置为评估模式
        
    # 遍历验证数据
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        
        images, bboxes = batch[0], batch[1]
        
        # 验证阶段不需要计算梯度，节省内存和加速
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            
            #----------------------#
            #   注意：验证阶段不需要清零梯度
            #   因为已经使用了 torch.no_grad()，不会计算梯度
            #----------------------#
            
            #----------------------#
            #   前向传播：模型推理
            #----------------------#
            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs, bboxes)

        # 累加验证损失
        val_loss += loss_value.item()
        
        # 更新进度条
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    # ==================== 记录和保存 ====================
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        # 记录损失到历史记录（并更新可视化）
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        
        # 执行模型评估（计算 mAP）
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        
        # 打印训练信息
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存模型权重
        #-----------------------------------------------#
        # 选择要保存的模型状态（优先使用 EMA 模型）
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # 定期保存检查点（每 save_period 个 epoch 或最后一个 epoch）
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                save_state_dict, 
                os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                    epoch + 1, 
                    loss / epoch_step, 
                    val_loss / epoch_step_val
                ))
            )
            
        # 保存最佳模型（验证损失最低的模型）
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # 始终保存最新模型（用于恢复训练）
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))