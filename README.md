# Block-gs
## 安装

```bash
conda create -n block_gs python=3.11 -y
conda activate block_gs
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

#如果 修改了cuda部分，需要重新编译
pip install submodules/submodules/hierarchy-rasterizer

```

**注意**以下是几个submodules的渲染cuda代码说明
1. hierarchy-rasterizer 基于原版hier_gs。
2. diff-gaussian-rasterization 基于rade_gs。
3. dom-rasterizer 用于生成正射影像，不用于训练 在`render`文件夹下为高斯生成正射影像代码。


## 训练

### 利用球训练
加入参数`--use_sphere` 以及 `--sphere_to_ellipsoid_iter 15000`设置从球变成椭球的迭代次数 
```
python train_single.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --use_sphere --sphere_to_ellipsoid_iter 15000
```

### train_single 天空球 scaffold 调用说明20240824
- 为了后续步骤 注意使用`--sh_degree 0`
- 若**加载scaffold**(其中包含了天空壳)，按照最初的层级Guass读取即可，在读取scaffold的过程中会在`pc_info.txt`下读取天空球的点数
```
--skybox_locked --scaffold_file <scaffold_path> --bounds_file <bounds file>

# ex
--skybox_locked --scaffold_file D:\Dataset\hier_ex\hqb_data\\output/scaffold/point_cloud/iteration_30000 --bounds_file D:\Dataset\hier_ex\hqb_data\\camera_calibration\\chunks\\5_5
```
- **不加载scaffold但需要天空壳**，在train_single指定了只训练颜色和不透明度，直接给定数目即可
```
--skybox_num 100000
```

### 控制包围壳外的分裂比例
`--percent_outside 0.1` 表示外部满足分裂条件的高斯点中只有百分之十会进行分裂

### Normal path 法向量
`-n` stable normal模型输出的法向量
`--normal_from_iter` 为法向量加入的迭代轮数
若使用`--auto_iteration` 默认在 0.3 * all_iterations 开始加入

### rade gs restriction 
<<<<<<< HEAD
`--regularization_from_iter` 默认为 0.3 * all_iterations

### progress_path
输入`--progress_path`，会在此文件夹下方生成名为progress的文件,一个int数(0-100)表示完成train_single的比例
=======
`regularization_from_iter` 默认为 0.3 * all_iterations
>>>>>>> test-rade
