import torch
import yaml
import os
from policy import ACTPolicy
from imitate_episodes import make_policy 

def export_act_to_onnx(
    ckpt_path,
    output_path,
    camera_names,
    chunk_size=8,
    hidden_dim=512,
    dim_feedforward=2048,
    kl_weight=1,
):
    """将 ACT 模型从 .ckpt 导出为 ONNX 文件"""
    # 构造 policy_config
    policy_config = {
        'num_queries': 100,          
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'kl_weight': 10,
        'enc_layers': 4,            
        'dec_layers': 7,             
        'nheads': 8,                
        'camera_names': ['main'],    
        'lr': 1e-5,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
    }

    # 构建模型并加载权重
    policy = ACTPolicy(policy_config)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(state_dict, strict=True)
    policy.eval().cuda()

    # 构造假输入
    # qpos: [B, state_dim], image: [B, N_cam, C, H, W]
    B, state_dim = 1, 14
    N_cam, C, H, W = len(camera_names), 3, 480, 640
    dummy_qpos = torch.randn(B, state_dim).cuda()
    dummy_image = torch.randn(B, N_cam, C, H, W).cuda()

    # 导出 ONNX
    torch.onnx.export(
        policy,
        (dummy_qpos, dummy_image),
        output_path,
        input_names=['qpos', 'images'],
        output_names=['actions'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'qpos': {0: 'batch'},
            'images': {0: 'batch'},
            'actions': {0: 'batch'},
        },
    )

    print(f"[EXPORT_INFO] 成功导出 ONNX 模型到: {output_path}")


if __name__ == "__main__":
    export_act_to_onnx(
        ckpt_path="ckpt/with_temp/policy_best.ckpt",
        output_path="ckpt/with_temp/policy_act.onnx",
        camera_names=["top"],  
        chunk_size=8,
        hidden_dim=512,
        dim_feedforward=2048,
    )
    
'''
 python export.py \
    --ckpt_dir ckpt/with_temp \
    --policy_class ACT \
    --task_name sim_transfer_cube_scripted \
    --seed 0 \
    --num_epochs 10000 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --kl_weight 10 \
    --camera_names main
'''