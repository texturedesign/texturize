neural-texturize
================

Automatically generate new textures similar to your source image.  Useful if you
want to make variations on a theme or expand the size of an existing texture.

1. Examples & Usage
===================

The main script takes a source image as a texture, and generates a new output that
captures the style of the original.  Here are some examples:

.. code-block:: bash

    texturize samples/grass.webp --size=1440x960 --output=result.png
    texturize samples/gravel.png --iterations=200 --precision=1e-6
    texturize samples/sand.tiff  --output=tmp/sand-{scale}.webp


For details about the command-line options, see the tool itself:

.. code-block:: bash

    texturize --help

Here are the command-line options currently available::

    Usage:
        texturize SOURCE [-s WxH] [-o FILE]
                         [--scales=S] [--precision=P] [--iterations=I]
        texturize --help

    Options:
        SOURCE                  Path to source image to use as texture.
        -s WxH, --size=WxH      Output resolution as WIDTHxHEIGHT. [default: 640x480]
        --scales=S              Number of scales to process. [default: 5]
        --precision=P           Set the quality for the optimization. [default: 1e-5]
        --iterations=I          Maximum number of iterations each octave. [default: 99]
        -o FILE, --output=FILE  Filename for saving the result. [default: {source}_gen.png]
        -h --help               Show this message.


2. Installation
===============

This repository uses submodules, so you'll need to clone it recursively to ensure
dependencies are available:

.. code-block:: bash

    git clone --recursive https://github.com/photogeniq/neural-texturize.git

Then, you can create a new virtual environment called ``myenv`` by installing
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and calling the following
commands:

.. code-block:: bash

    cd neural-texturize
    conda env create -n myenv -f tasks/setup-cuda.yml

Once the virtual environment is created, you can activate it and finish the setup of
``neural-texturize`` with these commands:

.. code-block:: bash

    conda activate myenv
    poetry install

Finally, you can check if everything worked by calling the script:

.. code-block:: bash

    texturize

You can use ``conda env remove`` to delete the virtual environment once you are done.
