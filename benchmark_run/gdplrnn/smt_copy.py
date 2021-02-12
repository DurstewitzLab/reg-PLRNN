#!/usr/bin/env python

from sumatra.projects import load_project
import shutil
import os.path
import sys
import click


def copy_record(project_source, project_destination, label, move=False):
    project_destination.add_record(project_source.get_record(label))
    src_path = os.path.join(project_source.path,
                            project_source.data_store.archive_store,
                            '{}.tar.gz'.format(label))
    dest_path = os.path.join(project_destination.path,
                             project_destination.data_store.archive_store,
                             '{}.tar.gz'.format(label))
    shutil.copy2(src_path, dest_path)
    print('copied {} from {} to {}'.format(label, project_source.name,
                                           project_destination.name))
    if move:
        project_source.delete_record(label, delete_data=True)


@click.command()
@click.option(
    '--move', is_flag=True, help='if provided move records instead of copying')
@click.argument('src', nargs=1, type=click.Path(exists=True))
@click.argument('dest', nargs=1, type=click.Path(exists=True))
@click.argument('labels', nargs=-1)
def cli(src, dest, labels, move):
    project_source = load_project(src)
    project_destination = load_project(dest)
    if len(labels) > 0:
        for label in labels:
            copy_record(project_source, project_destination, label, move)
    else:
        source_labels = project_source.record_store.labels(project_source.name)
        destination_labels = project_destination.record_store.labels(
            project_destination.name)
        for label in source_labels:
            if label not in destination_labels:
                copy_record(project_source, project_destination, label, move)


if __name__ == '__main__':
    cli()
