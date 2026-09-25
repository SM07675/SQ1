# SatQuery building model

SatlasPretrain Aerial Swin-v2-Base + footprint/boundary decoder, fine-tuned on SpaceNet 2.

- Train AOIs: Vegas, Shanghai
- Validation/tuning AOI: Paris
- Held-out test AOI: Khartoum
- Test instance F1 @ IoU 0.5: 0.3166
- Test semantic IoU: 0.5640
- Test count MAE: 13.617
- Test mean relative count error: 0.6988

Limitations: validate on target SatQuery geographies/sensors. Off-nadir displacement, tiny structures, dense touching roofs, clouds, and domain shift can degrade counts. Use overlap blending and one global watershed pass for large scenes.
