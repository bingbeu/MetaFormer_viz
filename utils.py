import os
import torch
import importlib
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def relative_bias_interpolate(checkpoint,config):
    for k in list(checkpoint['model']):
        if 'relative_position_index' in k:
            del checkpoint['model'][k]
        if 'relative_position_bias_table' in k:
            relative_position_bias_table = checkpoint['model'][k]
            cls_bias = relative_position_bias_table[:1,:]
            relative_position_bias_table = relative_position_bias_table[1:,:]
            size = int(relative_position_bias_table.shape[0]**0.5)
            img_size = (size+1)//2
            if 'stage_3' in k:
                downsample_ratio = 16
            elif 'stage_4' in k:
                downsample_ratio = 32
            new_img_size = config.DATA.IMG_SIZE//downsample_ratio
            new_size = 2*new_img_size-1
            if new_size == size:
                continue
            relative_position_bias_table = relative_position_bias_table.reshape(size,size,-1)
            relative_position_bias_table = relative_position_bias_table.unsqueeze(0).permute(0,3,1,2)#bs,nhead,h,w
            relative_position_bias_table = torch.nn.functional.interpolate(
                relative_position_bias_table, size=(new_size, new_size), mode='bicubic', align_corners=False)
            relative_position_bias_table = relative_position_bias_table.permute(0,2,3,1)
            relative_position_bias_table = relative_position_bias_table.squeeze(0).reshape(new_size*new_size,-1)
            relative_position_bias_table = torch.cat((cls_bias,relative_position_bias_table),dim=0)
            checkpoint['model'][k] = relative_position_bias_table
    return checkpoint
    
    
def load_pretained(config,model,logger=None,strict=False):
    if logger is not None:
        logger.info(f"==============> pretrain form {config.MODEL.PRETRAINED}....................")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    if config.MODEL.DORP_HEAD:
        if 'head.weight' in checkpoint['model'] and 'head.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"==============> drop head....................")
            del checkpoint['model']['head.weight']
            del checkpoint['model']['head.bias']
        if 'head.fc.weight' in checkpoint['model'] and 'head.fc.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"==============> drop head....................")
            del checkpoint['model']['head.fc.weight']
            del checkpoint['model']['head.fc.bias']
    if config.MODEL.DORP_META:
        if logger is not None:
            logger.info(f"==============> drop meta head....................")
        for k in list(checkpoint['model']):
            if 'meta' in k:
                del checkpoint['model'][k]
            
    checkpoint = relative_bias_interpolate(checkpoint,config)
    if 'point_coord' in checkpoint['model']:
        if logger is not None:
            logger.info(f"==============> drop point coord....................")
        del checkpoint['model']['point_coord']
    msg = model.load_state_dict(checkpoint['model'], strict=strict)
    del checkpoint
    torch.cuda.empty_cache()


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    max_accuracy = 0.0

    # 只有"恢复训练"才需要 optimizer / lr_scheduler / epoch。
    # 可视化/推理传 None 时，整段直接跳过，不再崩。
    can_resume = (
        not config.EVAL_MODE
        and 'optimizer' in checkpoint
        and 'lr_scheduler' in checkpoint
        and 'epoch' in checkpoint
        and optimizer is not None
        and lr_scheduler is not None
    )

    if can_resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if lr_scheduler is not None:
            new_base = config.TRAIN.BASE_LR
            for pg in optimizer.param_groups:
                pg['lr'] = new_base
            if hasattr(lr_scheduler, 'base_values'):
                lr_scheduler.base_values = [new_base] * len(lr_scheduler.base_values)
            if hasattr(lr_scheduler, 'lr_min'):
                lr_scheduler.lr_min = config.TRAIN.MIN_LR
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()

        # amp 可选加载：config 键可能缺失/非 CfgNode，amp 可能为 None，都要防
        cfg = checkpoint.get('config', None)
        cfg_amp = getattr(cfg, 'AMP_OPT_LEVEL', None) if cfg is not None else None
        if ('amp' in checkpoint
                and amp is not None
                and config.AMP_OPT_LEVEL != "O0"
                and cfg_amp != "O0"):
            amp.load_state_dict(checkpoint['amp'])

        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

_BEST_ACC = -1.0
def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    global _BEST_ACC          # ← 加这一行，告诉 Python 用的是模块级变量
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    latest_save_path = os.path.join(config.OUTPUT, 'latest.pth')
    logger.info(f"{latest_save_path} saving......")
    torch.save(save_state, latest_save_path)
    logger.info(f"{latest_save_path} saved !!!")

    # ---- 新增：只保存历史最优 best.pth ----
    if max_accuracy > _BEST_ACC:
        _BEST_ACC = max_accuracy
        best_save_path = os.path.join(config.OUTPUT, 'best.pth')
        torch.save(save_state, best_save_path)
        logger.info(f"{best_save_path} saved as BEST (max_accuracy {max_accuracy:.4f}) !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt




def load_ext(name, funcs):
    ext = importlib.import_module(name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext

