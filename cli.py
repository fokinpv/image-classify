#!/usr/bin/env python3
import click

from lib import run


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
@click.argument('img_path')
def classify(img_path):
    click.echo(f'classify image "{img_path}"')


if __name__ == '__main__':
    cli()
