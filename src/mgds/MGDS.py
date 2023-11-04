import gc
import random
from abc import abstractmethod, ABCMeta
from random import Random
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset


class PipelineModule(metaclass=ABCMeta):
    pipeline: 'LoadingPipeline'

    __base_seed: int
    __module_index: int

    __item_cache_index: int
    __item_cache: dict
    __length_cache: int

    def __init__(self):
        self.clear_item_cache()

    def init(self, pipeline: 'LoadingPipeline', base_seed: int, module_index: int):
        self.pipeline = pipeline

        self.__base_seed = base_seed
        self.__module_index = module_index

    def clear_item_cache(self):
        self.__item_cache_index = -1
        self.__item_cache = {}
        self.__length_cache = -1

    def get_previous_item(self, name: str, index: int):
        split_name = name.split('.')
        item_name = split_name[0]
        path_names = split_name[1::]

        item = None

        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if item_name in module.get_outputs():
                # item is cached
                if module.__item_cache_index == index and item_name in module.__item_cache.keys():
                    item = module.__item_cache[item_name]

                # the wrong index is cached, clear cache and recalculate
                elif module.__item_cache_index != index:
                    item = module.get_item(index, item_name)
                    module.__item_cache_index = index
                    module.__item_cache = item
                    item = item[item_name]

                # the item is cached and the index is correct, but the item_name is not part of the cache
                # recalculate and add to the cache
                elif item_name not in module.__item_cache.keys():
                    item = module.get_item(index, item_name)
                    module.__item_cache.update(item)
                    item = item[item_name]

                # if the item was found, break the loop
                # else, fall through to a previous module and try again
                if item is not None:
                    break

        for path_name in path_names:
            if path_name in item:
                item = item[path_name]
            else:
                item = None
                break

        return item

    def get_previous_length(self, name: str):
        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                if module.__length_cache < 0:
                    module.__length_cache = module.length()
                return module.__length_cache

    def get_previous_meta(self, name: str):
        for previous_module_index in range(self.__module_index - 1, -1, -1):
            module = self.pipeline.modules[previous_module_index]
            if name in module.get_outputs():
                return module.get_meta(name)

    def _get_rand(self, index: int = -1) -> Random:
        seed = hash((self.__base_seed, self.__module_index, self.pipeline.current_epoch, index))
        return Random(seed)

    def _torch_gc(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def get_inputs(self) -> list[str]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[str]:
        pass

    def start(self):
        """
        Called once when the dataset is created.
        """
        pass

    def start_next_epoch(self):
        """
        Called once before each epoch, starting with the first epoch.
        """
        pass

    def start_next_step(self):
        """
        Called once before each step.
        If gradient accumulation is used, it should not be called while accumulating gradients
        """
        pass

    def get_meta(self, name: str) -> Any:
        """
        Called to return meta information about this module.

        :param name: the requested meta key
        :return: meta information
        """
        return None

    @abstractmethod
    def get_item(self, index: int, requested_name: str = None) -> dict:
        """
        Called to return an item or partial item from this module.
        If `requested_name` is None, the entire item should be returned.
        If `requested_name` is a string, only the specified key needs to be returned,
        but the whole item can be returned if it improves performance to return everything at once.

        :param index: the item index to return
        :param requested_name: the requested item key
        :return: an item or partial item
        """
        pass


class SettingsPipelineModule(PipelineModule):
    def __init__(self, settings: dict):
        super(SettingsPipelineModule, self).__init__()
        self.settings = settings

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ['settings']

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {
            'settings': self.settings
        }


class ConceptPipelineModule(PipelineModule):
    def __init__(self, concepts: list[dict]):
        super(ConceptPipelineModule, self).__init__()
        self.concepts = concepts

    def length(self) -> int:
        return len(self.concepts)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return ['concept']

    def get_item(self, index: int, requested_name: str = None) -> dict:
        return {
            'concept': self.concepts[index]
        }


class OutputPipelineModule(PipelineModule):
    def __init__(self, names: list[str]):
        super(OutputPipelineModule, self).__init__()
        self.names = names

    def length(self) -> int:
        return self.get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        for name in self.names:
            item[name] = self.get_previous_item(name, index)

        return item


class RandomAccessNotSupported(Exception):
    def __init__(self, module):
        super().__init__(f"{type(module).__name__} does not support random access")


class LoadingPipeline:
    device: torch.device
    dtype: torch.dtype
    allow_mixed_precision: bool
    concepts: list[dict]
    settings: dict
    modules: list[PipelineModule]
    output_module: PipelineModule
    current_epoch: int
    last_initialized_epoch: int
    batch_size: int
    initial_epoch: int
    initial_epoch_sample: int

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            allow_mixed_precision: bool,
            concepts: list[dict],
            settings: dict,
            modules: list[PipelineModule],
            batch_size: int,
            seed: int,
            initial_epoch: int = 0,
            initial_epoch_sample: int = 0,
    ):
        self.device = device
        self.dtype = dtype
        self.allow_mixed_precision = allow_mixed_precision
        self.concepts = concepts
        self.settings = settings
        self.modules = list(filter(lambda x: x is not None, self.__flatten(modules)))
        for module in self.modules:
            if isinstance(module, OutputPipelineModule):
                self.output_module = module

        self.modules.insert(0, ConceptPipelineModule(self.concepts))
        self.modules.insert(1, SettingsPipelineModule(self.settings))
        for index, module in enumerate(self.modules):
            module.init(self, seed, index)

        self.batch_size = batch_size
        self.initial_epoch = initial_epoch
        self.initial_epoch_sample = initial_epoch_sample - (initial_epoch_sample % batch_size)

        self.current_epoch = -1
        self.last_initialized_epoch = -1

        self.current_sample_index = 0

    def __flatten(self, data: list | object) -> list:
        if isinstance(data, list):
            new_list = []
            for x in [self.__flatten(x) for x in data]:
                new_list.extend(x)
            return new_list
        else:
            return [data]

    def length(self) -> int:
        """
        Returns the exact length of a current epoch. This number can change between epochs.
        """
        if self.current_epoch == self.initial_epoch:
            # for the initial epoch, initial_epoch_sample defines the amount of samples to skip
            return max(0, self.output_module.length() - self.initial_epoch_sample)
        else:
            return self.output_module.length()

    def approximate_length(self) -> int:
        """
        Returns an approximated length of a full epoch.
        The number may not be exact, because the length can change between epochs.
        """
        return max(0, self.output_module.length())

    def start(self):
        """
        Called after initializing the pipeline.
        Can be used to add caching or other logic that should run once.
        """

        # Set current_epoch to 0 to simulate calling start() during the first epoch.
        self.current_epoch = 0
        for module_index in range(len(self.modules)):
            module = self.modules[module_index]
            module.start()

        # Set current_epoch to initial_epoch to simulate calling start_next_epoch() during the current epoch.
        self.current_epoch = self.initial_epoch
        for module_index in range(len(self.modules)):
            module = self.modules[module_index]
            module.start_next_epoch()

        self.last_initialized_epoch = self.current_epoch

        # Reset current_epoch to initial_epoch - 1, because the epoch has not yet started.
        # Calling start_next_epoch() once, will start the current epoch
        self.current_epoch = self.initial_epoch - 1

    def start_next_epoch(self):
        self.current_epoch += 1

        # if the last initialized epoch is not the previous epoch, no initialization is needed
        if self.last_initialized_epoch == self.current_epoch - 1:
            for module in self.modules:
                # At the start of each epoch, the previous cache is cleared.
                # This prevents duplicating samples when training on single images.
                module.clear_item_cache()
                module.start_next_epoch()

        self.last_initialized_epoch = self.current_epoch

        # for the initial epoch, initial_epoch_sample defines the amount of samples to skip
        if self.current_epoch == self.initial_epoch:
            self.current_sample_index = self.initial_epoch_sample
        else:
            self.current_sample_index = 0

    def start_next_step(self):
        for module in self.modules:
            module.start_next_step()

    def __get_item(self, index: int) -> dict:
        return self.output_module.get_item(index)

    def __next__(self):
        item = self.__get_item(self.current_sample_index)
        self.current_sample_index += 1
        return item


class MGDS(IterableDataset):
    device: torch.device
    dtype: torch.dtype
    allow_mixed_precision: bool
    loading_pipeline: LoadingPipeline

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            allow_mixed_precision: bool,
            concepts: list[dict],
            settings: dict,
            definition: [PipelineModule],
            batch_size: int,
            seed: int = 42,
            initial_epoch: int = 0,
            initial_epoch_sample: int = 0
    ):
        self.device = device
        self.dtype = dtype
        self.allow_mixed_precision = allow_mixed_precision
        seed = (random.randint(-(1 << 30), 1 << 30) if seed == -1 else seed)
        self.loading_pipeline = LoadingPipeline(device, dtype, allow_mixed_precision, concepts, settings, definition,
                                                batch_size, seed, initial_epoch, initial_epoch_sample)

        self.loading_pipeline.start()

    def __len__(self):
        return self.loading_pipeline.length()

    def __iter__(self):
        return self.loading_pipeline

    def approximate_length(self) -> int:
        return self.loading_pipeline.approximate_length()

    def start_next_epoch(self):
        self.loading_pipeline.start_next_epoch()

    def start_next_step(self):
        self.loading_pipeline.start_next_step()


class TrainDataLoader(DataLoader):
    def __init__(self, dataset: MGDS, batch_size):
        super(TrainDataLoader, self).__init__(dataset, batch_size=batch_size, drop_last=True)
