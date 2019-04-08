[An active texture-based digital atlas enables automated mapping of structures and markers across brains.](https://www.nature.com/articles/s41592-019-0328-8)
Yuncong Chen, Lauren E. McElvain, Alexander S. Tolpygo, Daniel Ferrante, Beth Friedman, Partha P. Mitra, Harvey J. Karten, Yoav Freund & David Kleinfeld. 
_Nature Methods_ (2019/3/11)


This toolkit is written in Python 2.7.2 and has been tested on a machine with Intel Xeon W5580 3.20GHz 16-core CPU, 128GB RAM and a Nvidia Titan X GPU, running Linux Ubuntu 16.04. 

A complete run-through of the following demo takes roughly 2 hours. In order to allow an user to walk through the pipeline in as little time as possible, this demo only used 3 sections. Such small number of sections is not adequate to obtain an accurate registration. To get reasonable results, at least 10 sections are recommended for each landmark.

## Viewer

This project has an associated viewer located [here](https://activebrainatlas.github.io/MouseBrainAtlas/) to view three examples of structure registration in the brainstem region.

## Installation

#### Install CUDA (refer to [this page](https://mxnet.apache.org/versions/master/install/ubuntu_setup.html#cuda-dependencies))

```bash
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run`
sudo chmod +x cuda_9.0.176_384.81_linux-run
sudo ./cuda_9.0.176_384.81_linux-run
```
- Select "no" to “Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?”.
- Then download cuDNN (latest version for CUDA 9.0)

```bash
tar xvzf cudnn-9.0-linux-x64-v7.4.2.24.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
```

#### Install other non-python packages

- Install ImageMagick 6.8.9. `sudo apt-get install imagemagick`
- To use GUIs, install PyQt4 into the virtualenv according to [this answer](https://stackoverflow.com/a/28850104).
    - Install python-qt4 globaly: `sudo apt-get install python-qt4`
    - Create symbolic link of PyQt4 to your virtual env: `ln -s /usr/lib/python2.7/dist-packages/PyQt4/ mousebrainatlas_virtualenv/lib/python2.7/site-packages/`
    - Create symbolic link of sip.so to your virtual env: `ln -s /usr/lib/python2.7/dist-packages/sip.x86_64-linux-gnu.so mousebrainatlas_virtualenv/lib/python2.7/site-packages/`

#### Install python packages

A configuration script is provided to create a [virtualenv](https://virtualenv.pypa.io/en/stable/) called **mousebrainatlas-virtualenv** and install necessary packages.

- Change `REPO_DIR`, `ROOT_DIR`, `DATA_ROOTDIR`, `THUMBNAIL_DATA_ROOTDIR` in `setup/config.sh`
- The default `requirements.txt` assumes CUDA version of 9.0. If your CUDA version (check using `nvcc -V` or `cat /usr/local/cuda/version.txt`) is 9.1, replace `mxnet-cu90` with `mxnet-cu91` in `requirements.txt`. If your machine does not have a GPU, replace `mxnet-cu90` with `mxnet`. Refer to [official mxnet page](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU) for available pips.
- `source setup/config.sh`. Make sure we are now working under the `mousebrainatlas_virtualenv` virtual environment.
- `cd demo`.


## Preprocessing

Note that the `DEMO998_input_spec.ini` files for most steps are different and must be manually created according to the actual input. In the following instructions, "create `DEMO998_input_spec.ini` as (prep_id, version, resolution)" means using the same set of image names as `image_name_list` but set the `prep_id`, `version` and `resolution` accordingly.

- **Download demo data**. Run `python download_demo_data.py` to download necessary data. 

Make sure the folder content looks like:

```bash
├── brains_info
│   └── DEMO998.ini
├── jp2_files
│   └── DEMO998
│       ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_lossless.jp2
│       ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_lossless.jp2
│       └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_lossless.jp2
├── mxnet_models
│   └── inception-bn-blue
│       ├── inception-bn-blue-0000.params
│       ├── inception-bn-blue-symbol.json
│       └── mean_224.npy
└── CSHL_data_processed
│   └── DEMO998
│       └── DEMO998_sorted_filenames.txt
│       └── DEMO998_prep2_sectionLimits
└── operation_configs
    ├── crop_orig_template.ini
    ├── from_aligned_to_none.ini
    ├── from_aligned_to_padded.ini
    ├── from_none_to_aligned_template.ini
    ├── from_none_to_padded.ini
    ├── from_none_to_wholeslice.ini
    ├── from_padded_to_none.ini
    ├── from_padded_to_brainstem_template.ini
    ├── from_padded_to_wholeslice_template.ini
    └── from_wholeslice_to_brainstem.ini
```

- **Convert raw images from JPEG2000 to tif**. Edit `DEMO998_raw_input_spec.json`. Set `data_dirs`, `filepath_to_imageName_mapping` and `imageName_to_filepath_mapping`. Run `python jp2_to_tiff.py DEMO998 DEMO998_raw_input_spec.json`.

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_raw
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw.tif
```

- **Extract Neurotrace-blue channel**. Modify `DEMO998_input_spec.ini` as (None,None,raw). `python extract_channel.py example_specs/DEMO998_input_spec.ini 2 Ntb`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       └── DEMO998_raw_Ntb
│           ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_Ntb.tif
│           ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_Ntb.tif
│           └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_Ntb.tif
```
- **Rescale to thumbnail**. Modify `DEMO998_input_spec.ini` as (None,Ntb,raw). `python rescale.py example_specs/DEMO998_input_spec.ini thumbnail -f 0.03125`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       └── DEMO998_thumbnail_Ntb
│           ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_thumbnail_Ntb.tif
│           ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_thumbnail_Ntb.tif
│           └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_thumbnail_Ntb.tif
```

- **Global intensity normalization**. Modify `DEMO998_input_spec.ini` as (None,Ntb,thumbnail). `python normalize_intensity.py example_specs/DEMO998_input_spec.ini NtbNormalized`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       └── DEMO998_thumbnail_NtbNormalized
│           ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_thumbnail_NtbNormalized.tif
│           ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_thumbnail_NtbNormalized.tif
│           └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_thumbnail_NtbNormalized.tif
```

- **Create an ordered list of images**. Create `$DATA_ROOTDIR/CSHL_data_processed/DEMO998/DEMO998_sorted_filenames.txt`. This file should already be included in the initial download. Each row of the file contains an image name and its index. The file should look like:

```bash
MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242 225
MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250 230
MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257 235
```

- **Align images in this stack**. 
    - Copy operation config template `cp operation_configs/from_none_to_aligned_template.ini CSHL_data_processed/DEMO998/DEMO998_operation_configs/from_none_to_aligned.ini`. Modify `from_none_to_aligned.ini`. In particular make sure `elastix_parameter_fp` is valid. 
    - Modify `DEMO998_input_spec.ini` as (None,NtbNormalized,thumbnail). `python align_compose.py example_specs/DEMO998_input_spec.ini --op from_none_to_aligned`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       └── DEMO998_transformsTo_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250.csv
│       ├── DEMO998_elastix_output
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_to_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242
│       │   │   ├── elastix.log
│       │   │   ├── IterationInfo.0.R0.txt
│       │   │   ├── IterationInfo.0.R1.txt
│       │   │   ├── IterationInfo.0.R2.txt
│       │   │   ├── IterationInfo.0.R3.txt
│       │   │   ├── IterationInfo.0.R4.txt
│       │   │   ├── IterationInfo.0.R5.txt
│       │   │   ├── result.0.tif
│       │   │   └── TransformParameters.0.txt
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_to_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250
│       │       ├── elastix.log
│       │       ├── IterationInfo.0.R0.txt
│       │       ├── IterationInfo.0.R1.txt
│       │       ├── IterationInfo.0.R2.txt
│       │       ├── IterationInfo.0.R3.txt
│       │       ├── IterationInfo.0.R4.txt
│       │       ├── IterationInfo.0.R5.txt
│       │       ├── result.0.tif
│       │       └── TransformParameters.0.txt
```

- **Transform images**. `python warp_crop.py --input_spec example_specs/DEMO998_input_spec.ini --op_id from_none_to_padded --njobs 8`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep1_thumbnail_NtbNormalized
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep1_thumbnail_NtbNormalized.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep1_thumbnail_NtbNormalized.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep1_thumbnail_NtbNormalized.tif
```

- Make sure the `all_stacks` meta-variable in `src/utilities/metadata.py` includes `DEMO998`.

- **Generate masks**. The masks should be included in the initial download. For how to generate them from scratch. refer to [this page](doc/mask_generation.md).

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep1_thumbnail_mask
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep1_thumbnail_mask.png
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep1_thumbnail_mask.png
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep1_thumbnail_mask.png
│       ├── DEMO998_thumbnail_mask
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_thumbnail_mask.png
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_thumbnail_mask.png
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_thumbnail_mask.png
```

- **Local adaptive intensity normalization**. Modify `DEMO998_input_spec.ini` as (None,Ntb,raw). `python normalize_intensity_adaptive.py input_spec.ini NtbNormalizedAdaptiveInvertedGamma`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_intensity_normalization_results
│       │   ├── floatHistogram
│       │   │   ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_floatHistogram.png
│       │   │   ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_floatHistogram.png
│       │   │   └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_floatHistogram.png
│       │   ├── meanMap
│       │   │   ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_meanMap.bp
│       │   │   ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_meanMap.bp
│       │   │   └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_meanMap.bp
│       │   ├── meanStdAllRegions
│       │   │   ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_meanStdAllRegions.bp
│       │   │   ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_meanStdAllRegions.bp
│       │   │   └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_meanStdAllRegions.bp
│       │   ├── normalizedFloatMap
│       │   │   ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_normalizedFloatMap.bp
│       │   │   ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_normalizedFloatMap.bp
│       │   │   └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_normalizedFloatMap.bp
│       │   ├── regionCenters
│       │   │   ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_regionCenters.bp
│       │   │   ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_regionCenters.bp
│       │   │   └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_regionCenters.bp
│       │   └── stdMap
│       │       ├── DEMO998_MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_raw_stdMap.bp
│       │       ├── DEMO998_MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_raw_stdMap.bp
│       │       └── DEMO998_MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_raw_stdMap.bp
```

- **Whole-slice crop**.  
    - Copy operation config template `cp $DATA_ROOTDIR/operation_configs/from_padded_to_wholeslice_template.ini $DATA_ROOTDIR/CSHL_data_processed/DEMO998/DEMO998_operation_configs/from_padded_to_wholeslice.ini`. Modify `from_padded_to_wholeslice.ini`. In this file specify the cropbox for the domain `alignedWithMargin` based on `alignedPadded` images. 
    - Modify `DEMO998_input_spec.ini` as (None,NtbNormalizedAdaptiveInvertedGamma,raw). `python warp_crop.py --input_spec example_specs/DEMO998_input_spec.ini --op_id from_none_to_wholeslice`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep5_raw_NtbNormalizedAdaptiveInvertedGamma
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep5_raw_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep5_raw_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep5_raw_NtbNormalizedAdaptiveInvertedGamma.tif
```

- Modify `input_spec.ini` as (alignedWithMargin,NtbNormalizedAdaptiveInvertedGamma,raw). `python rescale.py example_specs/DEMO998_input_spec.ini thumbnail -f 0.03125`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
```

- **Brainstem crop**. 
    - Copy operation config template `cp operation_configs/from_padded_to_brainstem_template.ini CSHL_data_processed/DEMO998/DEMO998_operation_configs/from_padded_to_brainstem.ini`. Modify `from_padded_to_brainstem.ini`.
    - Modify `DEMO998_input_spec.ini` as (alignedWithMargin,NtbNormalizedAdaptiveInvertedGamma,raw). `python warp_crop.py --input_spec example_specs/DEMO998_input_spec.ini --op_id from_wholeslice_to_brainstem`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep2_raw_NtbNormalizedAdaptiveInvertedGamma
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_raw_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_raw_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_raw_NtbNormalizedAdaptiveInvertedGamma.tif
```

- **Generate thumbnails**. Modify `DEMO998_input_spec.ini` as (alignedBrainstemCrop, NtbNormalizedAdaptiveInvertedGamma, raw). `python rescale.py example_specs/DEMO998_input_spec.ini thumbnail -f 0.03125`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep2_thumbnail_NtbNormalizedAdaptiveInvertedGamma
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_thumbnail_NtbNormalizedAdaptiveInvertedGamma.tif
```

- **Compress JPEG**. Use the same `DEMO998_input_spec.ini` as previous step. `python compress_jpeg.py example_specs/DEMO998_input_spec.ini`

```bash
├── CSHL_data_processed
│   └── DEMO998
│       ├── DEMO998_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg
│       │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg.jpg
│       │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg.jpg
│       │   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg.jpg
```
 
- **Identify section limits that only contain the brainstem**. Create `$DATA_ROOTDIR/CSHL_data_processed/DEMO998/DEMO998_prep2_sectionLimit.ini`. This file should already be included in the initial download. The content looks like:

```bash
[DEFAULT]
left_section_limit = 225
right_section_limit = 235
```

## Registration

- **Download atlas**. Run `python download_atlas.py`.

```bash
├── CSHL_volumes
│   ├── atlasV7
│   │   └── atlasV7_10.0um_scoreVolume
│   │       └── score_volumes
│   │           ├── atlasV7_10.0um_scoreVolume_12N.bp
│   │           ├── atlasV7_10.0um_scoreVolume_12N_origin_wrt_canonicalAtlasSpace.txt
│   │           ├── atlasV7_10.0um_scoreVolume_12N_surround_200um.bp
│   │           ├── atlasV7_10.0um_scoreVolume_12N_surround_200um_origin_wrt_canonicalAtlasSpace.txt
│   │           ├── atlasV7_10.0um_scoreVolume_3N_R.bp
│   │           ├── atlasV7_10.0um_scoreVolume_3N_R_origin_wrt_canonicalAtlasSpace.txt
│   │           ├── atlasV7_10.0um_scoreVolume_3N_R_surround_200um.bp
│   │           ├── atlasV7_10.0um_scoreVolume_3N_R_surround_200um_origin_wrt_canonicalAtlasSpace.txt
│   │           ├── atlasV7_10.0um_scoreVolume_4N_R.bp
│   │           ├── atlasV7_10.0um_scoreVolume_4N_R_origin_wrt_canonicalAtlasSpace.txt
│   │           ├── atlasV7_10.0um_scoreVolume_4N_R_surround_200um.bp
│   │           └── atlasV7_10.0um_scoreVolume_4N_R_surround_200um_origin_wrt_canonicalAtlasSpace.txt
```

- Compute **rough global transform**. The relevant results should already be included in the initial download. For details on how to obtain it from scratch, follow the instructions on [this page](doc/rough_global_registration.md).

```bash
├── CSHL_simple_global_registration
│   ├── DEMO998_registered_atlas_structures_wrt_wholebrainXYcropped_xysecTwoCorners.json
│   └── DEMO998_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.txt
```

- **Download pre-trained classifiers**. Run `python download_pretrained_classifiers.py -s "[\"12N\", \"3N\", \"4N\"]"`.

```bash
├── CSHL_classifiers
│   └── setting_899
│       └── classifiers
│           ├── 12N_clf_setting_899.dump
│           ├── 3N_clf_setting_899.dump
│           └── 4N_clf_setting_899.dump
```

- **Generate 3-D probability maps**. Run `python generate_prob_volumes.py DEMO998 799 NtbNormalizedAdaptiveInvertedGamma NtbNormalizedAdaptiveInvertedGammaJpeg -s "[\"12N\", \"3N\", \"4N\"]"`.

```bash
├── CSHL_patch_features
│   └── inception-bn-blue
│       └── DEMO998
│           └── DEMO998_prep2_none_win7
│               ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_none_win7_inception-bn-blue_features.bp
│               ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_none_win7_inception-bn-blue_locations.txt
│               ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_none_win7_inception-bn-blue_features.bp
│               ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_none_win7_inception-bn-blue_locations.txt
│               ├── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_none_win7_inception-bn-blue_features.bp
│               └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_none_win7_inception-bn-blue_locations.txt
├── CSHL_scoremaps
│   └── 10.0um
│       └── DEMO998
│           └── DEMO998_prep2_10.0um_detector799
│               ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_detector799
│               │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_detector799_12N_scoremap.bp
│               │   ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_detector799_3N_scoremap.bp
│               │   └── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_detector799_4N_scoremap.bp
│               ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_detector799
│               │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_detector799_12N_scoremap.bp
│               │   ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_detector799_3N_scoremap.bp
│               │   └── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_detector799_4N_scoremap.bp
│               └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_detector799
│                   ├── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_detector799_12N_scoremap.bp
│                   ├── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_detector799_3N_scoremap.bp
│                   └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_detector799_4N_scoremap.bp
├── CSHL_scoremap_viz
│   └── 10.0um
│       ├── 12N
│       │   └── DEMO998
│       │       └── detector799
│       │           └── prep2
│       │               ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_12N_detector799_scoremapViz.jpg
│       │               ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_12N_detector799_scoremapViz.jpg
│       │               └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_12N_detector799_scoremapViz.jpg
│       ├── 3N
│       │   └── DEMO998
│       │       └── detector799
│       │           └── prep2
│       │               ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_3N_detector799_scoremapViz.jpg
│       │               ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_3N_detector799_scoremapViz.jpg
│       │               └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_3N_detector799_scoremapViz.jpg
│       └── 4N
│           └── DEMO998
│               └── detector799
│                   └── prep2
│                       ├── MD662&661-F81-2017.06.06-12.44.40_MD661_2_0242_prep2_10.0um_4N_detector799_scoremapViz.jpg
│                       ├── MD662&661-F84-2017.06.06-14.03.51_MD661_1_0250_prep2_10.0um_4N_detector799_scoremapViz.jpg
│                       └── MD662&661-F86-2017.06.06-14.56.48_MD661_2_0257_prep2_10.0um_4N_detector799_scoremapViz.jpg
├── CSHL_volumes
│   └── DEMO998
│       ├── DEMO998_detector799_10.0um_scoreVolume
│       │   ├── score_volume_gradients
│       │   │   ├── DEMO998_detector799_10.0um_scoreVolume_12N_gradients.bp
│       │   │   ├── DEMO998_detector799_10.0um_scoreVolume_12N_origin_wrt_wholebrain.txt
│       │   │   ├── DEMO998_detector799_10.0um_scoreVolume_3N_R_gradients.bp
│       │   │   ├── DEMO998_detector799_10.0um_scoreVolume_3N_R_origin_wrt_wholebrain.txt
│       │   │   ├── DEMO998_detector799_10.0um_scoreVolume_4N_R_gradients.bp
│       │   │   └── DEMO998_detector799_10.0um_scoreVolume_4N_R_origin_wrt_wholebrain.txt
│       │   └── score_volumes
│       │       ├── DEMO998_detector799_10.0um_scoreVolume_12N.bp
│       │       ├── DEMO998_detector799_10.0um_scoreVolume_12N_origin_wrt_wholebrain.txt
│       │       ├── DEMO998_detector799_10.0um_scoreVolume_3N_R.bp
│       │       ├── DEMO998_detector799_10.0um_scoreVolume_3N_R_origin_wrt_wholebrain.txt
│       │       ├── DEMO998_detector799_10.0um_scoreVolume_4N_R.bp
│       │       └── DEMO998_detector799_10.0um_scoreVolume_4N_R_origin_wrt_wholebrain.txt
```
- **Register 12N (hypoglossal nucleus)**. Run `python register_brains.py example_specs/demo_fixed_brain_spec_12N.json example_specs/demo_moving_brain_spec_12N.json -g`.
- **Register 3N(oculomotor nucleus)/4N(trochlear nucleus) complex**. Run `python register_brains.py example_specs/demo_fixed_brain_spec_3N_R_4N_R.json example_specs/demo_moving_brain_spec_3N_R_4N_R.json -g`

```bash
├── CSHL_registration_parameters
│   └── atlasV7
│       ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N
│       │   ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_parameters.json
│       │   ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_scoreEvolution.png
│       │   ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_scoreHistory.bp
│       │   └── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_trajectory.bp
│       └── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R
│           ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_parameters.json
│           ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_scoreEvolution.png
│           ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_scoreHistory.bp
│           └── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_trajectory.bp
├── CSHL_volumes
│   ├── atlasV7
│   │   ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_10.0um
│   │   │   └── score_volumes
│   │   │       ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_10.0um_12N.bp
│   │   │       ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_10.0um_12N_origin_wrt_fixedWholebrain.txt
│   │   │       ├── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_10.0um_12N_surround_200um.bp
│   │   │       └── atlasV7_10.0um_scoreVolume_12N_warp7_DEMO998_detector799_10.0um_scoreVolume_12N_10.0um_12N_surround_200um_origin_wrt_fixedWholebrain.txt
│   │   ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um
│   │   │   └── score_volumes
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R.bp
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R_origin_wrt_fixedWholebrain.txt
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R_surround_200um.bp
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R_surround_200um_origin_wrt_fixedWholebrain.txt
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_4N_R.bp
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_4N_R_origin_wrt_fixedWholebrain.txt
│   │   │       ├── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_4N_R_surround_200um.bp
│   │   │       └── atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO998_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_4N_R_surround_200um_origin_wrt_fixedWholebrain.txt
│   │   └── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um
│   │       └── score_volumes
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_12N.bp
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_12N_origin_wrt_fixedWholebrain.txt
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_12N_surround_200um.bp
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_12N_surround_200um_origin_wrt_fixedWholebrain.txt
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_3N_R.bp
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_3N_R_origin_wrt_fixedWholebrain.txt
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_3N_R_surround_200um.bp
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_3N_R_surround_200um_origin_wrt_fixedWholebrain.txt
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_4N_R.bp
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_4N_R_origin_wrt_fixedWholebrain.txt
│   │           ├── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_4N_R_surround_200um.bp
│   │           └── atlasV7_10.0um_scoreVolume_warp0_DEMO998_detector799_10.0um_scoreVolume_10.0um_4N_R_surround_200um_origin_wrt_fixedWholebrain.txt
```

- **Visualize registration**. Run `python visualize_registration.py NtbNormalizedAdaptiveInvertedGamma example_specs/demo_visualization_per_structure_alignment_spec.json -g example_specs/demo_visualization_global_alignment_spec.json`

```bash
├── CSHL_registration_visualization
│   └── DEMO998_atlas_aligned_multilevel_down16_all_structures
│       └── NtbNormalizedAdaptiveInvertedGammaJpeg
│           ├── DEMO998_NtbNormalizedAdaptiveInvertedGammaJpeg_225.jpg
│           ├── DEMO998_NtbNormalizedAdaptiveInvertedGammaJpeg_230.jpg
│           └── DEMO998_NtbNormalizedAdaptiveInvertedGammaJpeg_235.jpg
```

An example registration visualization is given below (white contours = rough global registration, colored = different probability contours of locally registered structures)
![example registration_visualization](doc/example_registration_visualization.jpg?raw=true "Title")

