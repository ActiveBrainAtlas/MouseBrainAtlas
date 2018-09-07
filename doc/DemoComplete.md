To run any demo,
```
cd /home/yuncong/Brain/
source demo/set_env_variables.sh
```

## Preprocess 
- Run `download_demo_data_preprocessing.py --step 1` to download an example JPEG2000 image.
- Run `preprocess_demo.py --step 1`
- Run `download_demo_data_preprocessing.py --step 2`
- `$ ./align.py DEMO999 `
- 

## Compute patch features
- Run `demo/download_demo_data_compute_features.py`
- `$ ENABLE_UPLOAD_S3=0 ENABLE_DOWNLOAD_S3=0 ./demo/compute_features_demo.py DEMO999 --section 151 --version NtbNormalizedAdaptiveInvertedGamma`

#### Alex Running Notes
For DOWNLOAD DEMO DATA:
- Ran with `python demo/download_demo_data_compute_features.py --demo_data_dir /media/alexn/BstemAtlasDataBackup/demo/` to copy all files into the external hard drive. Only takes in 1 arg which is the download dir, otherwise defaults to ./demo_data. All files go into `[demo_data_dir]/CSHL_data_processed/DEMO999/`, Downloaded 4 files:
  - `DEMO999_sorted_filenames.txt`
  - `DEMO999_anchor.txt`
  - `DEMO999_alignedTo_MD662&661-F116-2017.06.07-04.39.41_MD661_1_0346_prep2_sectionLimits.json`
  - `DEMO999_alignedTo_MD662&661-F116-2017.06.07-04.39.41_MD661_1_0346_prep2_cropbox.json`
  
For COMPUTE FEATURE DEMO:
- Running it gives me the following error: `IOError: [Errno 2] No such file or directory: '/home/alexn/mxnet_models/inception-bn-blue/mean_224.npy'`. Attempting to fix
  - MXNET_MODEL_ROOTDIR set as ROOT_DIR/mxnet_models. I am changing that to be in BstemAtlasDataBackup 
    - Omitting several hours of errors I dealt with. Instead I'll problematic filepath selection elements:
      - Beginning of `distributed_utilities.py` is definition of `default_root` which is set as the beginning of the filepath where S3 files are downloaded. Needs to be changed
      - 
  - All problematic changes I implemented. Trying to make them work has already eaten up hours and hours. The specific problem is how learning_utilities.load_mxnet_model() is designed, it is used to download 'mxnet models' from S3 but does not accept a download filepath, instead it assumes where you want to put them based on what Yuncong hardcoded in the past. I will need to rewrite this function as well as a few others with similar problems for this to work.
 
## Generate probability volumes
- Run `demo/download_demo_data_scoring.py`
- `$ ENABLE_UPLOAD_S3=0 ENABLE_DOWNLOAD_S3=0 ./from_images_to_score_volumes_demo.py DEMO999 799 --structure_list [\"3N, 4N, 12N\"]`