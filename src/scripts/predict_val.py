import torch
import numpy as np
import pandas as pd
import sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.utils.utils import MyLightningCLI, TrainerWandb, TempSetContextManager


def predict_val(model_dirpath, output_dirpath):
    # Setup
    args = sys.argv[1:]
    with TempSetContextManager(sys, 'argv', sys.argv[:1]):
        cli = MyLightningCLI(
            trainer_class=TrainerWandb, 
            save_config_kwargs={
                'config_filename': 'config_pl.yaml',
                'overwrite': True,
            },
            args=[arg for arg in args if arg != 'fit'],
            run=False
        )
    cli.datamodule.setup()

    # Load model weights
    state_dict = torch.load(model_dirpath / f'last_{cli.datamodule.hparams.split_index}.ckpt')['state_dict']
    cli.model.load_state_dict(state_dict)
    cli.model = cli.model.cuda()
    cli.model.eval()

    # Create output directory
    output_dirpath.mkdir(parents=True, exist_ok=True)
    (output_dirpath / f'images_{cli.datamodule.hparams.split_index}').mkdir(parents=True, exist_ok=True)

    # Predict
    embeddings, dfs_meta, logits = [], [], []
    for batch in tqdm(cli.datamodule.val_dataloader()):
        with torch.no_grad():
            feats = cli.model.model.forward_features(batch['image'].cuda())
            logs = cli.model.model.head(feats)
            embs = cli.model.model.head.flatten(
                cli.model.model.head.global_pool(
                    feats
                )
            )
        
        # Save images
        for i in range(len(batch['meta']['eeg_id'])):
            eeg_id = batch['meta']['eeg_id'].iloc[i]
            eeg_sub_id = batch['meta']['eeg_sub_id'].iloc[i]
            img = batch['image'][i].cpu().numpy()

            # Save as png
            filepath = output_dirpath / f'images_{cli.datamodule.hparams.split_index}' / f'{eeg_id}_{eeg_sub_id}.png'
            img = (img * 255).astype(np.uint8)
            img = img[0]
            img = Image.fromarray(img)
            img.save(filepath)

        embeddings.append(embs.detach().cpu().numpy())
        logits.append(logs.detach().cpu().numpy())
        dfs_meta.append(batch['meta'])

    # Save
    embeddings = np.concatenate(embeddings)
    logits = np.concatenate(logits)
    dfs_meta = pd.concat(dfs_meta)
    dfs_meta['split_index'] = cli.datamodule.hparams.split_index

    np.save(output_dirpath / f'embeddings_{cli.datamodule.hparams.split_index}.npy', embeddings)
    np.save(output_dirpath / f'logits_{cli.datamodule.hparams.split_index}.npy', logits)
    dfs_meta.to_csv(output_dirpath / f'meta_{cli.datamodule.hparams.split_index}.csv', index=False)


def main():
    predict_val(
        model_dirpath=Path('models'),
        output_dirpath=Path('labels/predict_val')
    )


if __name__ == '__main__':
    main()
