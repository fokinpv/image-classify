#!/usr/bin/env python3
import click


@click.group()
def cli():
    pass


@cli.command(help='classify image')
@click.argument('image')
@click.option(
    '--checkpoint', default='checkpoint.pth',
    help='Model name'
)
def classify(image, checkpoint):
    from image_classify import recognize

    click.echo(f'classify image "{image}"')
    model, classes = recognize.load_checkpoint(checkpoint)
    guesses = recognize.do(image, model)
    for variant, probability in dict(guesses).items():
        print(f'{classes[variant]}: {probability}')


@cli.command(help="convert to jit script")
@click.argument('checkpoint')
def convert(checkpoint):
    import os.path

    import torch
    from image_classify import recognize

    click.echo(f'convert {checkpoint} to jit script')

    model, _ = recognize.load_checkpoint(checkpoint)
    model.eval()

    # An example input you would normally provide
    # to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    basename = os.path.basename(checkpoint)
    traced_script_module.save(f'traced_{basename}')


@cli.command(help='train model')
@click.argument('data_dir')
@click.option(
    '--model', default='densenet',
    help='Model name'
)
@click.option(
    '--epochs', default=25,
    help='Number of iteration in training process'
)
def train(data_dir, model, epochs):
    from image_classify import run

    click.echo(f'train model on data "{data_dir}"')
    run.start(data_dir, model, epochs)


if __name__ == '__main__':
    cli()
