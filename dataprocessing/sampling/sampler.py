

def check_label(tag: str):

    return 1 if "virus" in tag else 0


def sampling(seq: str, head: str, shift: int, win_size: int):

    # define output
    header, sequences, sequence_lengths, labels = [], [], [], []

    idx, ctr = 0, 1
    for i in range(0, len(seq) - win_size + 1, shift):
        ss = seq[i:i+win_size]

        header.append(head + f"_0{ctr}")
        sequences.append(ss)
        sequence_lengths.append(len(ss))
        labels.append(check_label(head))

        # get positions for the last sequence part
        idx = i + shift
        ctr += 1

    # add last part of the sequence
    last_ss = seq[idx:]
    header.append(head + f"_0{ctr}")
    sequences.append(last_ss)
    sequence_lengths.append(len(last_ss))
    labels.append(check_label(head))

    # save new data frame
    out_dict = {
        "header": header, "seqs": sequences, "seq_lens": sequence_lengths, "labels": labels
    }

    return out_dict


def cutting(seq: str, head: str, win_size: int):

    header, sequences, sequence_lengths, labels = [], [], [], []

    idx, ctr = 0, 1
    for i in range(0, len(seq) - win_size + 1, win_size):

        ss = seq[i:i+win_size] if i == 0 else seq[i-1:i+win_size]

        header.append(head + f"_0{ctr}")
        sequences.append(ss)
        sequence_lengths.append(len(ss))
        labels.append(check_label(head))

        # get positions for the last sequence part
        idx = i + win_size
        ctr += 1

    # add last part of the sequence
    last_ss = seq[idx-1:]
    header.append(head + f"_0{ctr}")
    sequences.append(last_ss)
    sequence_lengths.append(len(last_ss))
    labels.append(check_label(head))

    # save new data frame
    out_dict = {
        "header": header, "seqs": sequences, "seq_lens": sequence_lengths, "labels": labels
    }

    return out_dict
