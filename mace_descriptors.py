from pathlib import Path
from fire import Fire
from tqdm import tqdm

from ase.io import read

import torch
from mace import data
from mace.calculators import mace_mp
from mace.tools import torch_geometric, utils
from mace.modules.utils import extract_invariant
from mace.tools.scatter import scatter_sum


def mace_descriptors(
    data_path: str,
    save_path: str,
    batch_size: int = 32,
    device: str = "cuda",
    invariants_only=True,
    num_layers=-1,
):
    # set up the data path
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist.")
    # set up the save path
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    # set up the file paths
    raw_file_paths = sorted(data_path.glob("*.extxyz"))
    # omit the save path if it exists
    file_paths = [
        file_path
        for file_path in raw_file_paths
        if not (Path(save_path) / file_path.name.replace(".extxyz", ".pt")).exists()
    ]
    print(
        f"total: {len(raw_file_paths)}, "
        f"processed: {len(raw_file_paths) - len(file_paths)}, "
        f"remaining: {len(file_paths)}"
    )
    # set up the mace calculator
    mace_calc = mace_mp(default_dtype="float64", device=device)
    model = mace_calc.models[0]

    if num_layers == -1:
        num_layers = int(model.num_interactions)

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    for file_path in tqdm(file_paths):
        # set up the dataset and data loader
        atoms_list = read(file_path, index=":")
        config_list = [data.config_from_atoms(atoms) for atoms in atoms_list]
        data_list = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            for config in config_list
        ]
        data_loader = torch_geometric.dataloader.DataLoader(
            data_list, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # collect mace descriptors for each file path
        collector = []
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            node_features = model(batch.to_dict())["node_feats"]

            if invariants_only:
                irreps_out = model.products[0].linear.__dict__["irreps_out"]
                l_max = irreps_out.lmax
                num_features = irreps_out.dim // (l_max + 1) ** 2
                node_features = extract_invariant(
                    node_features,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )

            graph_features = scatter_sum(
                node_features, batch.batch, dim=0, dim_size=batch.num_graphs
            )  # [B, H]

            collector.append(graph_features)
        collector = torch.cat(collector, dim=0)
        torch.save(collector, save_path / file_path.name.replace(".extxyz", ".pt"))


if __name__ == "__main__":
    Fire(mace_descriptors)
