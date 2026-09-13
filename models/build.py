from timm.models import create_model  
from .MetaFG import *
from .MetaFG_meta import *


# 数据集 -> 类别名文本库路径（[num_classes, 768]）
_CATEGORY_BANK = {
    'cub-200': '/raid/datasets/cub-200/category_embeddings.npy',
    'nabirds': '/raid/datasets/nabirds/nabirds/category_embeddings.npy',
    'stanfordcars': '/raid/datasets/stanfordcars/category_embeddings.npy',
    'aircraft': '/raid/datasets/aircraft/category_embeddings.npy',
    'inaturelist2018': '/raid/datasets/inaturelist2018/category_embeddings.npy',
    'inaturelist2021': '/raid/datasets/inaturelist2021/category_embeddings.npy',
}


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'MetaFG':
        model = create_model(
                config.MODEL.NAME,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES, 
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                img_size=config.DATA.IMG_SIZE,
                only_last_cls=config.MODEL.ONLY_LAST_CLS,
                extra_token_num=config.MODEL.EXTRA_TOKEN_NUM,
                meta_dims=config.MODEL.META_DIMS,
                category_emb_path=_CATEGORY_BANK.get(config.DATA.DATASET),
                enable_hnsd=config.MODEL.HNSD_ENABLE,
                hnsd_margin=config.MODEL.HNSD_MARGIN,
                assess=config.MODEL.assess
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
