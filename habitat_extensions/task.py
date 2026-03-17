import gzip
import json
import os
from typing import Dict, List, Optional, Union

import attr
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from omegaconf import DictConfig

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import DatasetConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="R2RVLNCE-v1")
class VLNDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.config = config
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)


@registry.register_dataset(name="RxRVLNCE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.episodes = []
        self.config = config
        self.annotation_roles = self.extract_roles_from_config(config)
        self.languages = config.languages

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.data_path.format(split=config.split, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.scenes_dir)

        if ALL_LANGUAGES_MASK not in config.languages:
            languages_to_load = set(config.languages)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]


    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.split
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def extract_roles_from_config(cls, config: DictConfig) -> List[str]:
        if ALL_ROLES_MASK in config.roles:
            return cls.annotation_roles
        assert set(config.roles).issubset(set(cls.annotation_roles))
        return config.roles

    # @classmethod
    # def check_config_paths_exist(cls, config: DictConfig) -> bool:
    #     return all(
    #         os.path.exists(
    #             config.DATA_PATH.format(split=config.SPLIT, role=role)
    #         )
    #         for role in cls.extract_roles_from_config(config)
    #     ) and os.path.exists(config.SCENES_DIR)

    # @staticmethod
    # def _scene_from_episode(episode: VLNEpisode) -> str:
    #     """Helper method to get the scene name from an episode.  Assumes
    #     the scene_id is formated /path/to/<scene_name>.<ext>
    #     """
    #     return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language


cs = ConfigStore.instance()

@dataclass
class R2RVLNCEDatasetConfig(DatasetConfig):
    type: str = "R2RVLNCE-v1" 
    split: str = "train"
    scenes_dir: str = "data/scene_datasets/"
    data_path: str = "data/RxR_VLNCE_v0/{split}/{split}_gt.json.gz"

@dataclass
class RxRVLNCEDatasetConfig(DatasetConfig):
    type: str = "RxRVLNCE-v1" 
    split: str = "train"
    scenes_dir: str = "data/scene_datasets/"
    roles: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    data_path: str = "data/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"

cs.store(
    group="habitat/dataset", 
    name="r2rvlnce_v1", 
    node=R2RVLNCEDatasetConfig
)
cs.store(
    group="habitat/dataset", 
    name="rxrvlnce_v1", 
    node=RxRVLNCEDatasetConfig
)