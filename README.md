# PJ_WMA_Color_Object_Tracking

## Configuration

```
conda env create -f conda.yml
```

```
conda env update -f conda.yml
```

```
conda activate pja-wma
```

## Application

To run the application execute:

```
python color_object_tracker.py --video_path <path-to-video-file.mp4> --hue_tolerance 10 --saturation_tolerance 10 --value_tolerance 10
```