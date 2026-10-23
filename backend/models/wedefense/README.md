# WeDefense audio detector

The API can use a WeDefense audio-deepfake model exported as TorchScript. It
requires the paired artifacts described in the upstream deployment guide:

- a frontend (`*_frontend.pt`) accepting waveform and waveform-length tensors;
- a backend/projection model (`*.zip`) accepting frontend features.

Export them from a WeDefense checkpoint and its matching configuration:

```bash
python wedefense/deploy/export_jit.py --config <config.yaml> --checkpoint <checkpoint.pt> --output_file final.zip
```

Place the resulting files outside source control, for example under this
directory, then configure:

```env
SOCIAL_DETECT_WEDEFENSE_AUDIO_FRONTEND_PATH=/app/models/wedefense/final_frontend.pt
SOCIAL_DETECT_WEDEFENSE_AUDIO_BACKEND_PATH=/app/models/wedefense/final.zip
SOCIAL_DETECT_WEDEFENSE_AUDIO_SPOOF_CLASS_INDEX=1
```

The detector is optional and remains unavailable until both paths exist. The
class index must match the model's bonafide/spoof label ordering (the usual
two-class exports use spoof index `1`).
