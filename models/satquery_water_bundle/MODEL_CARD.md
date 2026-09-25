# SatQuery water model

SatlasPretrain Sentinel-2 multispectral Swin-v2-Base + explicit spectral fusion head, fine-tuned on hand-labeled Sen1Floods11 with event-exclusive splits and shadow hard negatives.

- Test F1: 0.8343
- Test IoU: 0.7157
- Test precision: 0.9209
- Test recall: 0.7626
- Dark-land false-positive rate: 0.0060

Limitations: optical imagery cannot see through thick cloud. Validate on the target sensor, geography, season, and water definition. Use Sentinel-1 fusion for all-weather flood mapping.
