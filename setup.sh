apt-get update
apt-get install vim git -y
apt-get install ffmpeg libsm6 libxext6  -y
pip install --upgrade pip
pip install torchvision
pip install opencv-python
cd $2
git pull
python $1