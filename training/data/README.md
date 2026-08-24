# Data Staging

Store dataset metadata, lightweight manifests, and instructions here.
Avoid checking full datasets into the repository.

Suggested structure:

```text
training/data/
  manifests/
    image_train.csv
    image_val.csv
    video_train.csv
    video_val.csv
  external/
    README.md
```

## Recommended datasets

- Images / synthetic-image detection:
  - GRIP-UNINA evaluation setup
  - your own social-media validation captures
- Videos / face deepfakes:
  - DFDC
  - FaceForensics++
  - Celeb-DF
  - DF40

## Manifest format

Use CSV files with at least:

```csv
path,label,split,source,platform
path/to/file.jpg,0,train,real_camera,instagram
path/to/file2.jpg,1,val,midjourney,unknown
```

Labels:

- `0` = real
- `1` = fake / AI-generated / manipulated
