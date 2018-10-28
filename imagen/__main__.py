#!/usr/bin/env python3
r"""                         _   _
  _ __   ___ _   _ _ __ __ _| | (_)_ __ ___   __ _  __ _  ___ _ __
 | '_ \ / _ \ | | | '__/ _` | | | | '_ ` _ \ / _` |/ _` |/ _ \ '_ \
 | | | |  __/ |_| | | | (_| | | | | | | | | | (_| | (_| |  __/ | | |
 |_| |_|\___|\__,_|_|  \__,_|_| |_|_| |_| |_|\__,_|\__, |\___|_| |_|
                                                   |___/
"""
#
# Copyright (c) 2018, Alex J. Champandard
#
# Neural Imagen is free software: you can redistribute it and/or modify it under the terms of
# the GNU Affero General Public License version 3.  This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

import argparse


__version__ = '0.1'


# ---------------------------------------------------------------------------------------------
# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'


def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)


def warn(message, *lines):
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))


print("""{}    {}\nGeneration and synthesis of bitmap images powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}\n"""
      .format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))


# =============================================================================================
# Algorithm & Processing
# =============================================================================================
class Application:

    def __init__(self):
        pass

    def cmd_reconstruct(self, args):
        from .commands import reconstruct
        return reconstruct.main(args)

    def cmd_synthesize(self, args):
        from .commands import synthesize
        return synthesize.main(args)


def main():
    app = Application()
    parser = argparse.ArgumentParser(prog='imagen')
    subparsers = parser.add_subparsers()

    parse_recons = subparsers.add_parser('reconstruct')
    parse_recons.add_argument('content', type=str, help='Filename of input image.')
    parse_recons.add_argument('--with', dest='style', type=str, required=True, help='Image for inspiration.')
    parse_recons.add_argument('--iterations', type=int, default=4, help='Number of iterations.')
    parse_recons.add_argument('--passes', type=int, default=2, help='Number of passes.')
    parse_recons.set_defaults(func=app.cmd_reconstruct)

    parse_synthz = subparsers.add_parser('synthesize')
    parse_synthz.add_argument('style', type=str, help='Filename of input image.')
    parse_synthz.add_argument('--iterations', type=int, default=8, help='Number of iterations.')
    parse_synthz.add_argument('--octaves', type=int, default=5, help='Number of octaves.')
    parse_synthz.set_defaults(func=app.cmd_synthesize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
