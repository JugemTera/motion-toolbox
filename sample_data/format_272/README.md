# 272-Dimensional Motion Representation Sample Data

Each `.npy` file has shape `(frames, 272)` with layout described in
[docs/272_FORMAT_REFERENCE.md](../../docs/272_FORMAT_REFERENCE.md).

## Sources

- Public reference implementation: [Li-xingXiao/272-dim-Motion-Representation](https://github.com/Li-xingXiao/272-dim-Motion-Representation)
- You can also generate 272D motion from SMPL using this package’s converters.

## Optional download helper

See `scripts/download_272_samples_template.py` for a Hugging Face
``snapshot_download`` template — fill in ``HF_USER`` / ``DATASET_NAME``
once the public mirror is finalised. ``scripts/fetch_sample_data_template.sh``
does the same for the rest of ``sample_data/``.

## License

Files you place in this directory may come from different datasets—**retain and follow each dataset’s license**.
