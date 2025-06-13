conda create -n miso python=3.9
conda activate miso

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install ninja

pip install trimesh

pip install opencv-python

pip install tensorboard==2.14.0

pip install pandas

pip install tqdm

pip install matplotlib==3.5.1

pip install rich

pip install packaging

pip install scipy

pip install imageio

pip install lpips

pip install torch-ema

pip install PyMCubes

pip install numpy==1.24.4

pip install pysdf

pip install dearpygui

pip install open3d==0.19.0

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install evo
