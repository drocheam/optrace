import tqdm  # progress bar

from . import global_options


class ProgressBar:

    def __init__(self, text: str, steps: int, **kwargs):
        """
        progress bar wrapper class. Uses tqdm internally

        :param text: text to display at front of progressbar
        :param steps: number of steps/iterations
        :param kwargs: additional parameters to tqdm
        """

        if global_options.show_progress_bar:
            self.bar = tqdm.tqdm(desc=text, total=steps, disable=None, ascii=True,
                                 bar_format='{l_bar}\b |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', **kwargs)
        else:
            self.bar = None

    def update(self, condition: bool = True) -> None:
        """
        Increment/update the state by one.
        
        :param condition: only update if condition is met
        """
        if self.bar is not None and condition:
            self.bar.update()

    def finish(self) -> None:
        """finish and close the progress bar"""
        if self.bar is not None:
            if (diff := self.bar.total - self.bar.n) > 0:
                self.update(diff)
            self.bar.close()

