class Scheduler:
    def __init__(self, optimizer):
        if not optimizer:
            raise ValueError("Optimizer must be provided.")
        self.optimizer = optimizer
        self.steps = 0

    def step(self, *args, **kwargs):
        raise NotImplementedError


class SequentialLR(Scheduler):
    def __init__(self, optimizer, schedulers: list[Scheduler], milestones: list[int] = []):
        """
        milestones: number of scheduler steps before switching to the next scheduler. If empty, the user must call next_scheduler() manually to switch schedulers.

        """
        super().__init__(optimizer)
        if not schedulers:
            raise ValueError("At least one scheduler must be provided.")
        if len(milestones) != len(schedulers) - 1 or len(milestones) != 0:
            raise ValueError(
                "Number of milestones must be one less than the number of schedulers. Or zero milestones should be given to switch schedulers manually."
            )

        self.schedulers = schedulers
        self.milestones = milestones
        self.cur_sch_idx = 0

    def _update_cur_sch_idx(self):
        if self.milestones:
            if self.cur_sch_idx < len(self.milestones) and self.steps >= self.milestones[self.cur_sch_idx]:
                self.cur_sch_idx += 1

    def step(self, *args, **kwargs):
        self.update_current_scheduler_idx()
        self.schedulers[self.cur_sch_idx].step(*args, **kwargs)
        self.steps += 1

    def next_scheduler(self):
        if self.current_scheduler_index < len(self.schedulers) - 1:
            self.current_scheduler_index += 1
