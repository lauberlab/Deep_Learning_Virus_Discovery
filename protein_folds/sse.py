import os.path
import subprocess as sp
import numpy as np


class SSEModule:

    def __init__(self, target_dir, SSE_default: dict, expected_size: int, SSE_code: dict = None,
                 enforce: bool = False):

        self.target_dir = target_dir
        self.expected_size = expected_size
        self.SSE_code = SSE_default if SSE_code is None else SSE_code
        self.enforce_execute = enforce

    @staticmethod
    def execute(command: list, header: str, tag: str):
        
        run = sp.Popen(command)
        run.wait()

        print(f"[SSE-DATABASE|{header}]: derived SSE based on '{tag}'.")

    def STRIDE_run(self, input_file, header):

        STRIDE_file = self.target_dir + header + ".stride"
        STRIDE_command = ["stride", input_file, f"-f{STRIDE_file}"]

        if not os.path.exists(STRIDE_file) or self.enforce_execute:

            self.execute(STRIDE_command, header, tag="STRIDE")

        else:
            print(f"[STRIDE]: @{header} found existing STRIDE file.")

        return STRIDE_file

    def DSSP_run(self, input_file, header):

        DSSP_file = self.target_dir + header + ".dssp"
        DSSP_command = ["dssp", "-i", input_file, "-o", DSSP_file]

        if not os.path.exists(DSSP_file) or self.enforce_execute:

            self.execute(DSSP_command, header, tag="DSSP")

        else:
            print(f"[DSSP]: @{header} found existing DSSP file.")

        return DSSP_file

    @staticmethod
    def read_file(file: str):

        with open(file, "r") as src:
            content = src.read().splitlines()

        return content

    def DSSP_read(self, file: str, isolates: bool = False):

        # read-in DSSP content
        DSSP_file_content = self.read_file(file=file)

        if len(DSSP_file_content) == 0:

            return None

        start_head = "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N" \
                     "    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA            CHAIN"

        # find start row
        sta_idx = DSSP_file_content.index(start_head) + 1

        # collect all necessary data entries
        DSSP_content = DSSP_file_content[sta_idx:]
        valid_entries = [i for i, line in enumerate(DSSP_content) if line[13:14] != "!"]
        col_ss = [line[16:17] for i, line in enumerate(DSSP_content) if i in valid_entries]

        # unify helices ('G', 'I' 'P' --> 'H')
        col_ss = ["H" if c == "G" or c == "I" or c == "P" else c for c in col_ss]

        # unify beta sheets ('B' --> 'E')
        col_ss = ["E" if c == "B" else c for c in col_ss]

        # unify coils/turns ('S' --> 'T)
        col_ss = ["T" if c == "S" else c for c in col_ss]

        # fill up all missing SSE as dummy coils 'C'
        col_ss = ["C" if cs == " " else cs for cs in col_ss]

        # curate isolated amino acid residues
        if isolates:

            # find all SSE that are 'alone' in terms of no consecutive SSE of the same type
            isolated_ss = [i for i, x in enumerate(col_ss) if
                           (i == 0 or col_ss[i - 1] != x) and (i == len(col_ss) - 1 or col_ss[i + 1] != x)]

            # iteration
            for iss in isolated_ss:

                # handling starting SSE
                if iss == 0:
                    col_ss[iss] = col_ss[iss + 1]
                    continue

                # and ending SSE
                if iss == len(col_ss) - 1:
                    col_ss[iss] = col_ss[iss - 1]
                    continue

                # in case the neighbouring SSE are identical to each other
                # the isolated SSE might be of the same type!
                if col_ss[iss - 1] == col_ss[iss + 1]:
                    col_ss[iss] = col_ss[iss + 1]

                # in case the isolated SSE is a coil 'C' or a turn 'T'
                # and the neighbouring SSE are either helices 'H' or strands 'E'
                # we assign either helix or beta sheet randomly
                if (col_ss[iss] == "T" or col_ss[iss] == "C") and (col_ss[iss - 1] == "H" or col_ss[iss + 1] == "H"):
                    col_ss[iss] = "H"

                if (col_ss[iss] == "T" or col_ss[iss] == "C") and (col_ss[iss - 1] == "E" or col_ss[iss + 1] == "E"):
                    col_ss[iss] = "E"

                # in case the isolated SSE is a turn 'T'
                # and the neighbouring SSE are either helices, strands or coils
                if col_ss[iss] == "T" and (col_ss[iss - 1] == "H" or col_ss[iss - 1] == "E") and col_ss[iss + 1] == "C":
                    col_ss[iss] = "C"

                # same in opposite direction
                if col_ss[iss] == "T" and col_ss[iss - 1] == "C" and (col_ss[iss + 1] == "H" or col_ss[iss + 1] == "E"):
                    col_ss[iss] = "C"

                # in case the isolated SSE is a coil 'C'
                # and the neighbouring SSE are either helices, strands or turns
                if col_ss[iss] == "C" and (col_ss[iss - 1] == "H" or col_ss[iss - 1] == "E") and col_ss[iss + 1] == "T":
                    col_ss[iss] = "T"

                # same in opposite direction
                if col_ss[iss] == "C" and col_ss[iss - 1] == "T" and (col_ss[iss + 1] == "H" or col_ss[iss + 1] == "E"):
                    col_ss[iss] = "T"

                # in case isolated SSE is either a helix 'H'
                # and the neighbouring SSE are either coils 'C' or turns 'T'
                if col_ss[iss] == "H" and (col_ss[iss - 1] == "T" or col_ss[iss + 1] == "T"):
                    col_ss[iss] = "T"

                if col_ss[iss] == "H" and (col_ss[iss - 1] == "C" or col_ss[iss + 1] == "C"):
                    col_ss[iss] = "C"

        # convert to SSE models code
        col_ss = [self.SSE_code[c] if c in self.SSE_code.keys() else (1, "C", "Coil") for c in col_ss]

        return col_ss

    def STRIDE_read(self, file: str):

        # get STRIDE file content
        STRIDE_content = self.read_file(file=file)
        STRIDE_content = [line for line in STRIDE_content if line[:3] == "ASG"]

        # find valid entries
        valid_entries = [i for i, line in enumerate(STRIDE_content) if line[13:14] != "!"]
        col_ss = [line[24:25] for i, line in enumerate(STRIDE_content) if i in valid_entries]

        # unify SSE notation using string comprehension
        col_ss_str = "".join(col_ss)
        col_ss_str = col_ss_str.replace("G", "H")
        col_ss_str = col_ss_str.replace("I", "H")
        col_ss_str = col_ss_str.replace("B", "E")
        col_ss_str = col_ss_str.replace("b", "E")

        # convert back to list
        col_ss = list(col_ss_str)

        # convert to SSE models code
        col_ss = [self.SSE_code[c] if c in self.SSE_code.keys() else (1, "C", "Coil") for c in col_ss]

        return col_ss

    """
    determine a consensus between STRIDE and DSSP based on the latter as priority
    meaning ss1=DSSP and ss2=STRIDE
    """
    def consensus(self, ss1, ss2):

        ss1_ids, ss1_ss = np.asarray([dd[0] for dd in ss1]), np.asarray([dd[1] for dd in ss1])
        ss2_ids, ss2_ss = np.asarray([dd[0] for dd in ss2]), np.asarray([dd[1] for dd in ss2])
        
        # size_check
        size_check = True if ss1_ss.shape[0] == self.expected_size and ss2_ss.shape[0] == self.expected_size else False

        if size_check:

            # save SSE pairs
            ss1_save = [(i, i + 1) for i, c in enumerate(ss1_ss[:-2]) if c != ss1_ss[i - 1] and c != ss1_ss[i + 2]]
            ss1_save = [index for pair in ss1_save for index in pair]

            # get all differences SSE-wise
            ss_diff = np.where(ss1_ss != ss2_ss)[0]

            for sdf in ss_diff:

                if sdf in ss1_save:
                    continue

                if (ss2_ss[sdf] == "H" or ss2_ss[sdf]) == "E" and ss1_ss[sdf] == "C":
                    ss1_ss[sdf] = ss2_ss[sdf]

        # convert back into tuple format
        ss1_ss = [self.SSE_code[c] if c in self.SSE_code.keys() else (1, "C", "Coil") for c in ss1_ss]

        return ss1_ss, size_check




