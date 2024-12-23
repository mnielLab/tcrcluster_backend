import torch
from torch.utils.data import Sampler
import numpy as np
import math
from itertools import cycle


class MinorityClassSampler(Sampler):
    """
        Custom Superclass to handle saving labels, minority classes, pepmap, indices as attributes as well as a counter
    """

    def __init__(self, labels, batch_size, minority_classes):
        super(MinorityClassSampler, self).__init__(labels)
        # self.pepmap = pepmap
        # self.inverse_pepmap = {v: k for k, v in pepmap.items()}
        self.labels = torch.tensor(labels)
        self.minority_classes = torch.tensor(minority_classes)
        self.n_minorities = len(minority_classes)
        self.non_minority_classes = torch.tensor(
            [cls for cls in torch.unique(self.labels) if cls not in minority_classes])
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(labels) / batch_size)
        self.last_size = len(labels) % batch_size
        self.counter = 0
        self.batch_count = 1
        self.available_indices = torch.full_like(self.labels, True)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        raise NotImplementedError

    def increment_counter(self):
        """
        Resets the batch counter and the available indices to indicate that one epoch has passed (i.e. all batches were done)
        Returns:

        """
        # self.class_cycle = cycle(self.minority_classes[torch.randperm(len(self.minority_classes))])
        self.counter += 1
        self.batch_count = 1
        self.available_indices.fill_(True)
        # print('Incremented!')


class GroupClassBatchSampler(MinorityClassSampler):
    def __init__(self, labels, batch_size, minority_classes):
        super(GroupClassBatchSampler, self).__init__(labels, batch_size, minority_classes)
        self.n_classes_to_pick = math.ceil(len(self.minority_classes) / self.num_batches)

    def __iter__(self):
        # Ensure reproducibility on each epoch
        torch.manual_seed(self.counter)
        # Shuffle minority classes each epoch so that they mix in together
        shuffled_minority_classes = self.minority_classes[torch.randperm(len(self.minority_classes))]

        for _ in range(self.num_batches):
            # Shuffled, select minimum between the n_classes to fit in a batch or the remaining clases
            chosen_classes = shuffled_minority_classes[:min(self.n_classes_to_pick, len(shuffled_minority_classes))]
            # shift in the shuffled classes vector to "advance"
            shuffled_minority_classes = shuffled_minority_classes[self.n_classes_to_pick:]
            batch_indices = []
            for cls in chosen_classes:
                class_indices = (self.labels == cls).nonzero(as_tuple=True)[0]
                count = min(len(class_indices), self.batch_size // 5)
                selected_minority = class_indices[:count]
                # This in theory shouldn't matter because we're gonna fit all of them "sequentially"
                self.available_indices[selected_minority] = False
                batch_indices.extend(selected_minority)

            # Fill the rest of the batch with samples from non-minority classes
            max_size = self.batch_size if self.batch_count != self.num_batches else self.last_size
            remaining_size = max_size - len(batch_indices)
            if remaining_size > 0:
                # Select all the majority_class indices that are ALSO available
                non_minority_indices = torch.cat(
                    [((self.labels == cls) & self.available_indices).nonzero(as_tuple=True)[0] \
                     for cls in self.non_minority_classes])
                # from all the indices ; Do randperm to shuffle, :remaning_size to index those up to a size or the maximum size
                non_minority_size = len(non_minority_indices)
                non_minority_indices = non_minority_indices[torch.randperm(non_minority_size)][
                                       :min(remaining_size, non_minority_size)]
                # Update available indices to keep track of what is available
                self.available_indices[non_minority_indices] = False
                batch_indices.extend(non_minority_indices)

            self.batch_count += 1
            yield torch.tensor(batch_indices)

        self.increment_counter()

    def __len__(self):
        return self.num_batches


# THIS IS not tested and doesn't work
class ClassSpecificBatchSampler(MinorityClassSampler):
    def __init__(self, pepmap, labels, batch_size, minority_classes, non_minority_classes):
        super(ClassSpecificBatchSampler, self).__init__(pepmap, labels, batch_size, minority_classes,
                                                        non_minority_classes)

        # Create a mapping from each minority class to its indices
        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in minority_classes}
        # Create a cycle of the class labels to rotate which class is used in each batch
        self.class_cycle = cycle(minority_classes)

    def __iter__(self):
        # Shuffle minority classes each epoch
        torch.manual_seed(torch.initial_seed())  # Ensure reproducibility on each epoch
        shuffled_minority_classes = torch.tensor(self.minority_classes)[torch.randperm(len(self.minority_classes))]

        for _ in range(self.num_batches):
            # Pick two random minority classes
            chosen_classes = shuffled_minority_classes[:2]  # Pick the first two after shuffle
            shuffled_minority_classes = torch.cat(
                (shuffled_minority_classes[2:], shuffled_minority_classes[:2]))  # Rotate

            batch_indices = []
            for cls in chosen_classes:
                class_indices = (self.labels == cls).nonzero(as_tuple=True)[0]
                class_indices = class_indices[torch.randperm(len(class_indices))]  # Shuffle indices
                # Ensure balanced class representation
                count = min(len(class_indices), self.batch_size // 4)
                batch_indices.extend(class_indices[:count])

            # Fill the rest of the batch with samples from non-minority classes
            remaining_size = self.batch_size - len(batch_indices)
            if remaining_size > 0:
                non_minority_indices = torch.cat(
                    [(self.labels == cls).nonzero(as_tuple=True)[0] for cls in self.non_minority_classes])
                non_minority_indices = non_minority_indices[torch.randperm(len(non_minority_indices))]
                batch_indices.extend(non_minority_indices[:remaining_size])
            batch_indices = torch.tensor(batch_indices)
            # Shuffle batch indices to mix classes
            batch_indices = batch_indices[torch.randperm(len(batch_indices))]
            yield batch_indices

    def __len__(self):
        return self.num_batches
