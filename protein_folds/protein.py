from typing import Any, Sequence, Mapping, Optional
import dataclasses
import numpy as np
import constants
import string


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def to_pdb(prot: Protein):

    restypes = constants.restypes + ["X"]
    res_1to3 = lambda r: constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    if np.any(aatype > constants.restype_num):
        raise ValueError("Invalid aatypes.")

    headers = get_pdb_headers(prot)

    if len(headers) > 0:
        pdb_lines.extend(headers)

    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase

    # outputs
    resids, residues, atoms, positions = [], [], [], []

    # Add all atom sites.
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])

        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            # numpy attributes
            positions.append(pos)
            atoms.append(atom_name)
            resids.append(residue_index[i])
            residues.append(constants.restype_Nto1[aatype[i]])
            # residues.append(res_name_3)

            # PDB string attributes
            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = (i == n - 1)
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if i != n - 1:
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines), np.asarray(atoms),\
        np.asarray(resids), np.asarray(residues), np.asarray(positions)


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [
            p for i, p in zip(parents_chain_index, parents) if i == chain_id
        ]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers
