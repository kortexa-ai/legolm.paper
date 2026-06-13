# Vision dataset

172 keyframes sampled evenly across **Big Buck Bunny** (596.46 s, 24 fps),
© 2008 Blender Foundation / www.bigbuckbunny.org, licensed **CC-BY 3.0**.
Source: https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov

Each `dataset.json` entry holds `frame_idx` (at 24 fps), `timestamp` (seconds),
`frame_path`, a caption generated with a locally hosted `gemma-4-12b`
vision-language model, and `token_ids` (the caption under the paper tokenizer,
truncated to 60 tokens).

The vision encoder checkpoint (`checkpoints/experiments/vision-perceiver.pt`)
was trained on these frame-caption pairs with the upstream legolm
`vision_train.py` (joint perceiver+LM caption objective, 50 epochs); the paper
runner uses only the frozen perceiver weights.
