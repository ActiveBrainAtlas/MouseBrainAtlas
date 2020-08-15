This page describes the location of the raw input and intermediate data.

The storage locations of different types of data are specified by the following global variables in `global_setting.py`:

- `REPO_DIR`: code repository
- `DATA_DIR`: processed images
- `VOLUME_ROOTDIR`: volumes
- `MESH_ROOTDIR`: 3-D meshes
- `REGISTRATION_PARAMETERS_ROOTDIR`: registration parameters
- `PATCH_FEATURES_ROOTDIR`: patch features
- `ANNOTATION_ROOTDIR`: annotation files
- `CLF_ROOTDIR`: classifiers
- `SCOREMAP_ROOTDIR`: score maps
- `SCOREMAP_VIZ_ROOTDIR`: JPG of score maps
- `REGISTRATION_VIZ_ROOTDIR`: registration visualization
- `MXNET_MODEL_ROOTDIR`: mxnet models

All data are stored relative to these root directories. The exact path for specific data is provided in the description of the system process that generates the data.

Data are stored long-term in two places:
- Lab server `birdstore.dk.ucsd.edu` under `/brainbucket/data/Active_Atlas_Data/`. 
  - Raw images are stored in folders that correspond to the scanner used, e.g. `UCSD_AxioScanner/CSHL2_2018-04-04`.
  - Processed images and other data are stored in `CSHL_data_processed` and other folders.
- AWS S3 bucket `mousebrainatlas-rawdata` and `mousebrainatlas-data`.
  - Raw images are stored in bucket `mousebrainatlas-rawdata`.
  - Processed images and other data are stored in bucket `mousebrainatlas-data`.

Refer to [this page](TransferFiles.md) on how to transfer data between lab server or S3 and the local workstation.

(## Reconstructed volumes or virtual sections
Collection of images representing virtual sections in all three directions (sagittal, coronal and horizontal).)

[List of brain stack directories](../Brain_stack_directories.md)
