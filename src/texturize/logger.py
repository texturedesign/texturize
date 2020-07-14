# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import progressbar

try:
    import ipywidgets
except ImportError:
    pass


class ansi:
    WHITE = "\033[1;97m"
    BLACK = "\033[0;30m\033[47m"
    YELLOW = "\033[1;33m"
    PINK = "\033[1;35m"
    ENDC = "\033[0m\033[49m"


class EmptyLog:
    def notice(self, *args):
        pass

    def info(self, *args):
        pass

    def debug(self, *args):
        pass

    def warn(self, *args):
        pass

    def create_progress_bar(self, iterations):
        return progressbar.NullBar(max_value=iterations)


class ConsoleLog:
    def __init__(self, quiet, verbose):
        self.quiet = quiet
        self.verbose = verbose

    def create_progress_bar(self, iterations):
        widgets = [
            progressbar.Variable("iter", format="{name}: {value}"),
            " | ",
            progressbar.Variable("loss", format="{name}: {value:0.3e}"),
            " ",
            progressbar.Bar(marker="■", fill="·"),
            " ",
            progressbar.Percentage(),
            " | ",
            progressbar.Timer(format='elapsed: %(elapsed)s'),
        ]
        ProgressBar = progressbar.NullBar if self.quiet else progressbar.ProgressBar
        return ProgressBar(
            max_value=iterations, widgets=widgets, variables={"loss": float("+inf")}
        )

    def debug(self, *args):
        if self.verbose:
            print(*args)

    def notice(self, *args):
        if not self.quiet:
            print(*args)

    def info(self, *args):
        if not self.quiet:
            print(ansi.BLACK + "".join(args) + ansi.ENDC)

    def warn(self, *args):
        print(ansi.YELLOW + "".join(args) + ansi.ENDC)


class NotebookLog:
    class ProgressBar:
        def __init__(self, max_iter):
            self.bar = ipywidgets.IntProgress(
                value=0,
                min=0,
                max=max_iter,
                step=1,
                description="",
                bar_style="",
                orientation="horizontal",
                layout=ipywidgets.Layout(width="100%", margin="0"),
            )

            from IPython.display import display

            display(self.bar)

        def update(self, value, **keywords):
            self.bar.value = value

        def reset(self, iterations):
            self.bar.max = iterations
            self.bar.value = 0
            self.bar.layout = ipywidgets.Layout(width="100%", margin="0")

        def finish(self):
            self.bar.layout = ipywidgets.Layout(display="none")

    def __init__(self):
        self.progress = None

    def create_progress_bar(self, iterations):
        if self.progress is None:
            self.progress = NotebookLog.ProgressBar(iterations)
        else:
            self.progress.reset(iterations)

        return self.progress

    def debug(self, *args):
        pass

    def notice(self, *args):
        pass

    def info(self, *args):
        pass

    def warn(self, *args):
        pass


def get_default_log():
    try:
        get_ipython
        ipywidgets
        return NotebookLog()
    except NameError:
        return EmptyLog()
