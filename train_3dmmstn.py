from argparse import ArgumentParser
from pytorch_lightning import Trainer
from lib.net_module import Net3DMMSTN


def hparam_parser():
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', default=1e-10, type=float)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--max_nb_epochs', '-epoch', default=1000, type=int)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--checkpoint_path', '-o', default='weights')
    parser.add_argument('--tutte_emb_path', '-te', default='models/model.mat')
    parser.add_argument(
        '--vgg_faces_path', '-vf', default='models/vgg_face_dag.pth')
    parser.add_argument(
        '--dataset_root', '-im', default='./data/aflw_processed_data')
    parser.add_argument(
        '--dataset_csv', '-csv', default='./data/aflw_cropped_label.csv')
    parser.add_argument('--worker', '-w', default=1)
    parser.add_argument('--dev_run', '-d', action='store_true', default=False)

    return parser


if __name__ == "__main__":
    hparams = hparam_parser().parse_args()

    model = Net3DMMSTN(hparams)
    trainer = Trainer(
        early_stop_callback=None,
        track_grad_norm=2,
        print_nan_grads=True,
        weights_summary='full',
        default_save_path=hparams.checkpoint_path,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        fast_dev_run=hparams.dev_run)
    trainer.fit(model)
