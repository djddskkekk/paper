import os.path as osp

from ..build import DATASET_REGISTRY
from .digits_dg import DigitsDG
from ..base_dataset import DatasetBase

from .oxford_pets import OxfordPets_dg


@DATASET_REGISTRY.register()
class OfficeHomeDG(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_fewshot_dir = osp.join(self.dataset_dir, "split_fewshot")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        print("+" * 100)
        print(type(cfg.DATASET.SOURCE_DOMAINS))
        print(cfg.DATASET.SOURCE_DOMAINS[0])
        domain1 = []
        domain2 = []
        domain3 = []
        domain1.append(cfg.DATASET.SOURCE_DOMAINS[0])
        domain2.append(cfg.DATASET.SOURCE_DOMAINS[1])
        domain3.append(cfg.DATASET.SOURCE_DOMAINS[2])
        # train = DigitsDG.read_data(
        #    self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "train"
        # )
        train1 = DigitsDG.read_data(
            self.dataset_dir, domain1, "train"
        )
        train2 = DigitsDG.read_data(
            self.dataset_dir, domain2, "train"
        )
        train3 = DigitsDG.read_data(
            self.dataset_dir, domain3, "train"
        )
        val = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "val"
        )
        test = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, "all"
        )

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = osp.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if osp.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                # train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                train1 = self.generate_fewshot_dataset(train1, num_shots=num_shots)
                train2 = self.generate_fewshot_dataset(train2, num_shots=num_shots)
                train3 = self.generate_fewshot_dataset(train3, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                # data = {"train": train, "val": val}
                # print(f"Saving preprocessed few-shot data to {preprocessed}")
                # with open(preprocessed, "wb") as file:
                #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(type(train1))

        print(train2["domain"])
        train = []
        train = train1 + train2 + train3
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets_dg.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
