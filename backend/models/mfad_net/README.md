# MFAD-Net checkpoint directory
#
# After smoke or full training:
#   cp training/exports/mfad_net/mfad_net_best.pth backend/models/mfad_net/
#
# Or set SOCIAL_DETECT_MFAD_NET_CHECKPOINT_PATH to an absolute path.
#
# Note: the committed smoke checkpoint is trained on *synthetic* trimodal
# data to validate the architecture end-to-end. Retrain with FaceForensics++,
# DFDC, and WildDeepfake (see training/mfad_net/README.md) before relying on
# it for real deepfake detection accuracy.
