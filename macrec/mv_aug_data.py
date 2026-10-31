"""mmgen addition: multi-view history augmentation for the chain decoder (docs/MFD_REPORT.md Part C).

Task `seqrec_mv` = MACRec's seqrec training examples, each held three times with the WHOLE history shown as one
view of every past item: full [f t i], text [f t] or image [f i] (mfd_data.VIEWS / view_maps). The target is
always the next item's full SID (one sequence, as in seqrec). No item-level translation tasks.
Validation stays `seqrec` (full view). The MFD counterpart is mfd_data.MVMFDDataset.
"""
from torch.utils.data import ConcatDataset

from data import SeqRecDataset
from mfd_data import VIEWS, view_maps

MV_TASKS = ("seqrec_mv",)


class MVSeqRecDataset(SeqRecDataset):

    def __init__(self, args):
        super().__init__(args, task="seqrec", mode="train")    # loads the full-SID index as seqrec does
        self.task = "seqrec_mv"
        self.vmap = view_maps(self.indices)

    def __len__(self):
        return len(VIEWS) * len(self.inter_data)

    def __getitem__(self, index):
        n = len(self.inter_data)
        d, view = self.inter_data[index % n], VIEWS[index // n]
        return dict(input_ids="".join(self.vmap[view][h] for h in d["inters"]), labels=d["item"],
                    label=index, task_flag=0)


def make_load_datasets(orig):
    """Wrap utils.load_datasets: seqrec_mv -> MVSeqRecDataset; other tasks unchanged."""
    def load(args):
        tasks = args.tasks.split(",")
        rest = [t for t in tasks if t.lower() not in MV_TASKS]
        mv = [MVSeqRecDataset(args) for t in tasks if t.lower() in MV_TASKS]
        if rest:
            saved = args.tasks
            args.tasks = ",".join(rest)
            train, valid = orig(args)
            args.tasks = saved
            return ConcatDataset(mv + list(train.datasets)), valid
        from data import SeqRecDataset as _S
        valid = ConcatDataset([_S(args, task=v, mode="valid") for v in args.valid_task.lower().split(",")])
        return ConcatDataset(mv), valid
    return load
