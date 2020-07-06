texturize
=========

.. image:: docs/gravel-x4.webp

A command-line tool and Python library to automatically generate new textures similar
to a source image or photograph.  It's useful in the context of computer graphics if
you want to make variations on a theme or expand the size of an existing texture.

This software is powered by deep learning technology — using a combination of
convolution networks and example-based optimization to synthesize images.  We're
building ``texturize`` as the highest-quality open source library available!

1. `Examples & Demos <#1-examples--demos>`_
2. `Commands <#2-commands>`_
3. `Options & Usage <#3-options--usage>`_
4. `Installation <#4-installation>`_

|Python Version| |License Type| |Project Stars|

.. image:: docs/grass-x4.webp


1. Examples & Demos
===================

The examples are available as notebooks, and you can run them directly in-browser
thanks to Jupyter and Google Colab:

* **Gravel** — `online demo <https://colab.research.google.com/github/photogeniq/neural-texturize/blob/master/examples/Demo_Gravel.ipynb>`__ and `source notebook <https://github.com/photogeniq/neural-texturize/blob/master/examples/Demo_Gravel.ipynb>`__.
* **Grass** — `online demo <https://colab.research.google.com/github/photogeniq/neural-texturize/blob/master/examples/Demo_Grass.ipynb>`__ and `source notebook <https://github.com/photogeniq/neural-texturize/blob/master/examples/Demo_Grass.ipynb>`__.

These demo materials are released under the Creative Commons `BY-NC-SA license <https://creativecommons.org/licenses/by-nc-sa/3.0/>`_, including the text, images and code.

2. Commands
===========

a) REMIX
--------

    Generate a variation of any shape from a single texture.

Remix Command-Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    Usage:
        texturize remix SOURCE...

    Examples:
        texturize remix samples/grass.webp --size=720x360
        texturize remix samples/gravel.png --size=512x512

Remix Library API
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from texturize import api, commands

    # The input could be any PIL Image in RGB mode.
    image = io.load_image_from_file("input.png")

    # Coarse-to-fine synthesis runs one octave at a time.
    remix = commands.Remix(image, mode="patch")
    for result in api.process_octaves(remix, octaves=5, threshold=1e-3):
        pass

    # The output can be saved in any PIL-supported format.
    result.image.save("output.png")


Remix Examples
~~~~~~~~~~~~~~

.. image:: docs/remix-gravel.webp

Remix Online Tool
~~~~~~~~~~~~~~~~~

* `colab notebook <https://colab.research.google.com/github/photogeniq/neural-texturize/blob/master/examples/Tool_Remix.ipynb>`__

----

b) REMAKE
---------

    Reproduce an original texture in the style of another.


Remake Command-Line
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    Usage:
        texturize remake TARGET like SOURCE

    Examples:
        texturize remake samples/grass1.webp like samples/grass2.webp
        texturize remake samples/gravel1.png like samples/gravel2.png


Remake Library API
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from texturize import api, commands

    # The input could be any PIL Image in RGB mode.
    image = io.load_image_from_file("input.png")

    # Only process one octave to retain photo-realistic output.
    remake = commands.Remake(image, mode="gram")
    for result in api.process_octaves(remake, octaves=1, threshold=1e-7):
        pass

    # The output can be saved in any PIL-supported format.
    result.image.save("output.png")



Remake Examples
~~~~~~~~~~~~~~~

.. image:: docs/remake-grass.webp

Remake Online Tool
~~~~~~~~~~~~~~~~~~

* `colab notebook <https://colab.research.google.com/github/photogeniq/neural-texturize/blob/master/examples/Tool_Remake.ipynb>`__

----

3. Options
==========

For details about the command-line options, see the tool itself:

.. code-block:: bash

    texturize --help

Here are the command-line options currently available::

    Usage:
        texturize SOURCE... [--size=WxH] [--output=FILE] [--variations=V] [--seed=SEED]
                            [--mode=MODE] [--octaves=O] [--threshold=H] [--iterations=I]
                            [--device=DEVICE] [--precision=PRECISION] [--quiet] [--verbose]

    Options:
        SOURCE                  Path to source image to use as texture.
        -s WxH, --size=WxH      Output resolution as WIDTHxHEIGHT. [default: 640x480]
        -o FILE, --output=FILE  Filename for saving the result, includes format variables.
                                [default: {source}_gen{variation}.png]
        --variations=V          Number of images to generate at same time. [default: 1]
        --seed=SEED             Configure the random number generation.
        --mode=MODE             Either "patch" or "gram" to specify critics. [default: gram]
        --octaves=O             Number of octaves to process. [default: 5]
        --threshold=T           Quality for optimization, lower is better. [default: 1e-4]
        --iterations=I          Maximum number of iterations each octave. [default: 99]
        --device=DEVICE         Hardware to use, either "cpu" or "cuda".
        --precision=PRECISION   Floating-point format to use, "float16" or "float32".
        --quiet                 Suppress any messages going to stdout.
        --verbose               Display more information on stdout.
        -h, --help              Show this message.


4. Installation
===============

If you're a developer and want to install the library locally, start by cloning the
repository to your local disk:

.. code-block:: bash

    git clone https://github.com/photogeniq/neural-texturize.git

Then, you can create a new virtual environment called ``myenv`` by installing
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and calling the following
commands, depending whether you want to run on CPU or GPU (via CUDA):

.. code-block:: bash

    cd neural-texturize

    # a) Use this if you have an *Nvidia GPU only*.
    conda env create -n myenv -f tasks/setup-cuda.yml

    # b) Fallback if you just want to run on CPU.
    conda env create -n myenv -f tasks/setup-cpu.yml

Once the virtual environment is created, you can activate it and finish the setup of
``neural-texturize`` with these commands:

.. code-block:: bash

    conda activate myenv
    poetry install

Finally, you can check if everything worked by calling the script:

.. code-block:: bash

    texturize

You can use ``conda env remove -n myenv`` to delete the virtual environment once you
are done.

----

|Python Version| |License Type| |Project Stars|

.. |Python Version| image:: https://img.shields.io/pypi/pyversions/texturize
    :target: https://www.python.org/

.. |License Type| image:: https://img.shields.io/badge/license-AGPL-blue.svg
    :target: https://github.com/photogeniq/neural-texturize/blob/master/LICENSE

.. |Project Stars| image:: https://img.shields.io/github/stars/photogeniq/neural-texturize.svg?style=flat
    :target: https://github.com/photogeniq/neural-texturize/stargazers
