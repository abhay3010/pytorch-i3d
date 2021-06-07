#!/bin/sh
apt-get update
# apt-get install vim git -y
# apt-get install ffmpeg libsm6 libxext6  -y
# pip install --upgrade pip
# # conda install torchvision
# pip install torchvision
# pip install opencv-python
# pip install sklearn
# pip install torchsummary
cd /mnt/data 
tar -xzf TinyViratWithClasses.tar.gz
#adding 
cp /mnt/data/TinyVIRAT/videos/train/VIRAT_S_000001/3759_frames/frame_75.jpg /mnt/data/TinyVIRAT/videos/train/VIRAT_S_000001/3759_frames/frame_76.jpg
cp /mnt/data/TinyVIRAT/videos/train/VIRAT_S_040104_05_000939_001116/6252_frames/frame_94.jpg /mnt/data/TinyVIRAT/videos/train/VIRAT_S_040104_05_000939_001116/6252_frames/frame_95.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_112.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_113.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_112.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_114.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_114.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040103_00_000000_000120/2165_frames/frame_115.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_104.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_105.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_105.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_106.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_106.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3410_frames/frame_107.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_108.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_109.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_109.jpg  /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_110.jpg
cp /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_110.jpg /mnt/data/TinyVIRAT/videos/test/VIRAT_S_040000_00_000000_000036/3413_frames/frame_111.jpg
cd $2
git pull
python $1
