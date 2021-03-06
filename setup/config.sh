#!/bin/bash

# User can modify this part
PROJECT_DIR=/home/yuncong/MouseBrainAtlas_dev_trial3
virtualenv="mousebrainatlas_virtualenv"
##################################################

red='\e[1;31m'
purple='\e[1;35m'
green='\e[1;32m'
cyan='\e[1;36m'
NC='\033[0m' # No Color

export REPO_DIR=$PROJECT_DIR/src

#export ROOT_DIR=$PROJECT_DIR/demo/demo_data/
#export DATA_ROOTDIR=$PROJECT_DIR/demo/demo_data/
#export THUMBNAIL_DATA_ROOTDIR=$PROJECT_DIR/demo/demo_data/

#export ROOT_DIR=/media/yuncong/BstemAtlasData/atlas_data
#export DATA_ROOTDIR=/media/yuncong/BstemAtlasData/atlas_data
#export THUMBNAIL_DATA_ROOTDIR=/media/yuncong/BstemAtlasData/atlas_data

export ROOT_DIR=/home/yuncong/demo_data
export DATA_ROOTDIR=/home/yuncong/demo_data
export THUMBNAIL_DATA_ROOTDIR=/home/yuncong/demo_data

#export ROOT_DIR=/home/yuncong/brainstem/media/yuncong/BstemAtlasData/atlas_data
#export DATA_ROOTDIR=/home/yuncong/brainstem/media/yuncong/BstemAtlasData/atlas_data
#export THUMBNAIL_DATA_ROOTDIR=/home/yuncong/brainstem/media/yuncong/BstemAtlasData/atlas_data

#export ROOT_DIR=/media/yuncong/YuncongPublic/atlas_data
#export DATA_ROOTDIR=/media/yuncong/YuncongPublic/atlas_data
#export THUMBNAIL_DATA_ROOTDIR=/media/yuncong/YuncongPublic/atlas_data

if [ ! -d $virtualenv ]; then
        echo ""
        echo -e "${green}Creating a virtualenv environment${NC}"
        #virtualenv --system-site-packages $virtualenv
        virtualenv -p python $PROJECT_DIR/$virtualenv
fi

echo ""
echo -e "${green}Activating the virtualenv environment${NC}"
source $PROJECT_DIR/$virtualenv/bin/activate

echo ""
echo -e "${green}[virtualenv] Installing Python packages${NC}"
pip install -r $PROJECT_DIR/setup/requirements.txt
