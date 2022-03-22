import lowrank.pruners


class AlignmentPruner(lowrank.pruners.Pruner):
    """
    Alignment pruners scores singular vectors based on how much each singular vector perturbs the
    model output from the baseline
    """

    def compute_masks(self) -> list[list[bool]]:
        pass  # TODO
