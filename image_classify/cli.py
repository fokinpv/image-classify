#!/usr/bin/env python3
import click

from . import run, predict


@click.group()
def cli():
    pass


@cli.command()
@click.argument('data_dir')
@click.option(
    '--model', default='densenet',
    help='Model name'
)
def train(data_dir, model):
    click.echo(f'train model on data "{data_dir}"')
    run.start(data_dir, model)


@cli.command()
@click.argument('image')
@click.option(
    '--checkpoint', default='checkpoint_ic_d161.pth',
    help='Model name'
)
def classify(image, checkpoint):
    click.echo(f'classify image "{image}"')
    guess = predict.do(image, checkpoint)
    print(guess)


if __name__ == '__main__':
    cli()
