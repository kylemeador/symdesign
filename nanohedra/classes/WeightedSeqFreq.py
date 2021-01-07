class FragMatchInfo:
    def __init__(self, res_pair_freqs, oligomer_1_ch_id, oligomer_1_res_num, oligomer_2_ch_id, oligomer_2_res_num,
                 z_val):
        self.res_pair_freqs = res_pair_freqs
        self.oligomer_1_ch_id = oligomer_1_ch_id
        self.oligomer_1_res_num = oligomer_1_res_num
        self.oligomer_2_ch_id = oligomer_2_ch_id
        self.oligomer_2_res_num = oligomer_2_res_num
        self.z_val = z_val

        self.score = 1 / float(1 + (z_val ** 2))

        self.oligomer_1_res_freqs = {}
        self.oligomer_2_res_freqs = {}
        for res_pair in self.res_pair_freqs:
            res1_type, res2_type = res_pair[0]
            res_freq = res_pair[1]

            if res1_type in self.oligomer_1_res_freqs:
                self.oligomer_1_res_freqs[res1_type] += res_freq
            else:
                self.oligomer_1_res_freqs[res1_type] = res_freq

            if res2_type in self.oligomer_2_res_freqs:
                self.oligomer_2_res_freqs[res2_type] += res_freq
            else:
                self.oligomer_2_res_freqs[res2_type] = res_freq

    def get_res_pair_freqs(self):
        return self.res_pair_freqs

    def get_oligomer_1_ch_id(self):
        return self.oligomer_1_ch_id

    def get_oligomer_1_res_num(self):
        return self.oligomer_1_res_num

    def get_oligomer_2_ch_id(self):
        return self.oligomer_2_ch_id

    def get_oligomer_2_res_num(self):
        return self.oligomer_2_res_num

    def get_z_val(self):
        return self.z_val

    def get_oligomer_1_res_freqs(self):
        return self.oligomer_1_res_freqs

    def get_oligomer_2_res_freqs(self):
        return self.oligomer_2_res_freqs

    def get_score(self):
        return self.score


class SeqFreqInfo:
    def __init__(self, frag_match_info_list):
        self.frag_match_info_list = frag_match_info_list

        self.oligomer_1 = []
        self.oligomer_2 = []

        oligomer_1_freqs_w_sum = {}
        oligomer_2_freqs_w_sum = {}
        score_sum_dict_1 = {}
        score_sum_dict_2 = {}
        for frag_match_info in frag_match_info_list:
            # get information from specific match
            match_score = frag_match_info.get_score()
            match_oligomer_1_ch_id = frag_match_info.get_oligomer_1_ch_id()
            match_oligomer_1_res_num = frag_match_info.get_oligomer_1_res_num()
            match_oligomer_1_freqs = frag_match_info.get_oligomer_1_res_freqs()
            match_oligomer_2_ch_id = frag_match_info.get_oligomer_2_ch_id()
            match_oligomer_2_res_num = frag_match_info.get_oligomer_2_res_num()
            match_oligomer_2_freqs = frag_match_info.get_oligomer_2_res_freqs()

            # weigh matched residue frequencies by match score
            # do so for the matched residue on oligomer 1 and the matched residue on oligomer 2
            match_oligomer_1_freqs_w = {res_type: freq * match_score
                                        for (res_type, freq) in match_oligomer_1_freqs.items()}
            match_oligomer_2_freqs_w = {res_type: freq * match_score
                                        for (res_type, freq) in match_oligomer_2_freqs.items()}

            # add match score to sum of the residue match scores
            # do so for the matched residue on oligomer 1 and the matched residue on oligomer 2
            if match_oligomer_1_ch_id in score_sum_dict_1:
                if match_oligomer_1_res_num in score_sum_dict_1[match_oligomer_1_ch_id]:
                    score_sum_dict_1[match_oligomer_1_ch_id][match_oligomer_1_res_num] += match_score
                else:
                    score_sum_dict_1[match_oligomer_1_ch_id][match_oligomer_1_res_num] = match_score
            else:
                score_sum_dict_1[match_oligomer_1_ch_id] = {match_oligomer_1_res_num: match_score}

            if match_oligomer_2_ch_id in score_sum_dict_2:
                if match_oligomer_2_res_num in score_sum_dict_2[match_oligomer_2_ch_id]:
                    score_sum_dict_2[match_oligomer_2_ch_id][match_oligomer_2_res_num] += match_score
                else:
                    score_sum_dict_2[match_oligomer_2_ch_id][match_oligomer_2_res_num] = match_score
            else:
                score_sum_dict_2[match_oligomer_2_ch_id] = {match_oligomer_2_res_num: match_score}

            # for each residue type add the weighted residue frequency to the sum of weighted residue type frequencies
            # do so for the matched residue on oligomer 1 and the matched residue on oligomer 2
            if match_oligomer_1_ch_id in oligomer_1_freqs_w_sum:
                if match_oligomer_1_res_num in oligomer_1_freqs_w_sum[match_oligomer_1_ch_id]:
                    for (match_res_type, match_freq) in match_oligomer_1_freqs_w.items():
                        if match_res_type in oligomer_1_freqs_w_sum[match_oligomer_1_ch_id][match_oligomer_1_res_num]:
                            oligomer_1_freqs_w_sum[match_oligomer_1_ch_id][match_oligomer_1_res_num][match_res_type] += match_freq
                        else:
                            oligomer_1_freqs_w_sum[match_oligomer_1_ch_id][match_oligomer_1_res_num][match_res_type] = match_freq
                else:
                    oligomer_1_freqs_w_sum[match_oligomer_1_ch_id][match_oligomer_1_res_num] = match_oligomer_1_freqs_w
            else:
                oligomer_1_freqs_w_sum[match_oligomer_1_ch_id] = {match_oligomer_1_res_num: match_oligomer_1_freqs_w}

            if match_oligomer_2_ch_id in oligomer_2_freqs_w_sum:
                if match_oligomer_2_res_num in oligomer_2_freqs_w_sum[match_oligomer_2_ch_id]:
                    for (match_res_type, match_freq) in match_oligomer_2_freqs_w.items():
                        if match_res_type in oligomer_2_freqs_w_sum[match_oligomer_2_ch_id][match_oligomer_2_res_num]:
                            oligomer_2_freqs_w_sum[match_oligomer_2_ch_id][match_oligomer_2_res_num][match_res_type] += match_freq
                        else:
                            oligomer_2_freqs_w_sum[match_oligomer_2_ch_id][match_oligomer_2_res_num][match_res_type] = match_freq
                else:
                    oligomer_2_freqs_w_sum[match_oligomer_2_ch_id][match_oligomer_2_res_num] = match_oligomer_2_freqs_w
            else:
                oligomer_2_freqs_w_sum[match_oligomer_2_ch_id] = {match_oligomer_2_res_num: match_oligomer_2_freqs_w}

        # for each residue calculate the weighted average frequency for the different residue types
        for ch_id_1 in oligomer_1_freqs_w_sum:
            for res_num_1 in oligomer_1_freqs_w_sum[ch_id_1]:
                res_score_sum_1 = score_sum_dict_1[ch_id_1][res_num_1]
                for res_type_1 in oligomer_1_freqs_w_sum[ch_id_1][res_num_1]:
                    oligomer_1_freqs_w_sum[ch_id_1][res_num_1][res_type_1] /= float(res_score_sum_1)

        for ch_id_2 in oligomer_2_freqs_w_sum:
            for res_num_2 in oligomer_2_freqs_w_sum[ch_id_2]:
                res_score_sum_2 = score_sum_dict_2[ch_id_2][res_num_2]
                for res_type_2 in oligomer_2_freqs_w_sum[ch_id_2][res_num_2]:
                    oligomer_2_freqs_w_sum[ch_id_2][res_num_2][res_type_2] /= float(res_score_sum_2)

        # sort sequence frequency information by chain ID, residue number and frequency
        # for oligomer 1 and oligomer 2
        for (ch_id_1, ch_id_1_resnums) in sorted(oligomer_1_freqs_w_sum.items(), key=lambda kv: kv[0]):
            sorted_freqs_1 = []
            for (res_num_1, res_num_1_freqs) in sorted(oligomer_1_freqs_w_sum[ch_id_1].items(), key=lambda kv: kv[0]):
                sorted_freqs_1.append((res_num_1, sorted(oligomer_1_freqs_w_sum[ch_id_1][res_num_1].items(), key=lambda kv: kv[1], reverse=True)))
            self.oligomer_1.append((ch_id_1, sorted_freqs_1))

        for (ch_id_2, ch_id_2_resnums) in sorted(oligomer_2_freqs_w_sum.items(), key=lambda kv: kv[0]):
            sorted_freqs_2 = []
            for (res_num_2, res_num_2_freqs) in sorted(oligomer_2_freqs_w_sum[ch_id_2].items(), key=lambda kv: kv[0]):
                sorted_freqs_2.append((res_num_2, sorted(oligomer_2_freqs_w_sum[ch_id_2][res_num_2].items(), key=lambda kv: kv[1], reverse=True)))
            self.oligomer_2.append((ch_id_2, sorted_freqs_2))

    def get_oligomer_1(self):
        return self.oligomer_1

    def get_oligomer_2(self):
        return self.oligomer_2

    def write(self, output_file_path):

        outfile = open(output_file_path, "a+")
        outfile.write("***** WEIGHTED RESIDUE FREQUENCIES *****\n\n")
        outfile.close()

        oligomers_seq_freqs = (self.get_oligomer_1(), self.get_oligomer_2())

        # write output for oligomers 1 and 2 to output file
        for oligomer in range(2):
            for (ch_id, ch_id_resnums) in oligomers_seq_freqs[oligomer]:
                outfile = open(output_file_path, "a+")
                outfile.write("OLIGOMER %s  |  CHAIN %s\n" % (str(oligomer + 1), ch_id))
                ch_id_res_freqs = []
                for (res_num, res_num_freqs) in ch_id_resnums:
                    res_num_freqs_rounded = [(res_type, round(res_freq, 3)) for (res_type, res_freq) in res_num_freqs]
                    outfile.write("RES NUM " + str(res_num) + "\n" + str(res_num_freqs_rounded) + "\n\n")
                    ch_id_res_freqs.append(res_num_freqs)
                outfile.write("\n")
                outfile.close()
