#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
    PIK3CA_mutation.cli
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    AI models for pathology images.

    :copyright: © 2019 by the Choppy Team.
    :license: AGPLv3+, see LICENSE for more details.
"""

"""Console script for PIK3CA_mutation."""


import click
import sys
from PIK3CA_mutation.prediction import start_models as start_prediction_models
from PIK3CA_mutation.single_prediction import start_model as start_prediction_model
from PIK3CA_mutation.single_heatmap import make_heatmap as start_single_heatmap
from PIK3CA_mutation.heatmap import make_heatmap as start_heatmap
from os.path import abspath, dirname
root_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, root_dir)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--sampling-file', '-f', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--seed', '-S', required=False,
              help="Random seed (default: 2020).", default=2020)
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--net', required=False, type=click.Choice(['resnet18', 'alexnet', 'resnet34',  'inception_v3']),
              help="Which net (default: resnet18)?", default='resnet18')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
@click.option('--num-workers', '-N', required=False,
              help="How many workers (default: 4)?", default=4, type=int)
@click.option('--batch-size', '-b', required=False,
              help="Batch size (default: 256)?", default=256, type=int)
def prediction(datapath, sampling_file, root_dir, seed, gpu, net, num_classes, num_workers, batch_size):
    """To predict with the specified model."""
    start_prediction_models(datapath, sampling_file=sampling_file, root_dir=root_dir, seed=seed,
                            gpu=gpu, net=net, num_classes=num_classes, num_workers=num_workers, batch_size=batch_size)


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--sampling-file', '-f', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--model-type', '-m', required=False,
              help="The model type for prediction (default: PIK3CA_Mutation).",
              default='PIK3CA_Mutation', type=click.Choice(['PIK3CA_Mutation', 'BLIS', 'IM', 'LAR',  'MES']))
@click.option('--seed', '-S', required=False,
              help="Random seed (default: 2020).", default=2020)
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--net', required=False, type=click.Choice(['resnet18', 'alexnet', 'resnet34',  'inception_v3']),
              help="Which net (default: resnet18)?", default='resnet18')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
@click.option('--num-workers', '-N', required=False,
              help="How many workers (default: 4)?", default=4, type=int)
@click.option('--batch-size', '-b', required=False,
              help="Batch size (default: 256)?", default=256, type=int)
def single_prediction(datapath, sampling_file, root_dir, model_type, seed, gpu, net, num_classes, num_workers, batch_size):
    """To predict with the specified model."""
    start_prediction_model(datapath, sampling_file=sampling_file, root_dir=root_dir, model_type=model_type,
                           seed=seed, gpu=gpu, net=net, num_classes=num_classes, num_workers=num_workers, batch_size=batch_size)


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--feats-file', '-f', required=True,
              help="The file which saved feats (npz file).",
              type=click.Path(exists=True, file_okay=True))
@click.option('--sampling-file', '-s', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--model-type', '-m', required=False,
              help="The model type for prediction (default: PIK3CA_Mutation).",
              default='PIK3CA_Mutation', type=click.Choice(['PIK3CA_Mutation', 'BLIS', 'IM', 'LAR',  'MES']))
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
def single_heatmap(datapath, feats_file, sampling_file, root_dir, model_type, gpu, num_classes):
    """To make a heatmap for the selected image patches."""
    start_single_heatmap(datapath, featsfile=feats_file, sampling_file=sampling_file,
                         root_dir=root_dir, model_type=model_type, gpu=gpu, num_classes=num_classes)


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--feats-file', '-f', required=True,
              help="The file which saved feats (npz file).",
              type=click.Path(exists=True, file_okay=True))
@click.option('--sampling-file', '-s', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
def heatmap(datapath, feats_file, sampling_file, root_dir, gpu, num_classes):
    """To make a heatmap for the selected image patches."""
    start_heatmap(datapath, featsfile=feats_file, sampling_file=sampling_file,
                  root_dir=root_dir, gpu=gpu, num_classes=num_classes)


main = click.CommandCollection(sources=[cli])

if __name__ == '__main__':
    main()
