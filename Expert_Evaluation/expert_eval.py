import json, copy, time, os

class ExpertEval(object):

    def __init__(self, exp_id) -> None:
        self.exp_id = exp_id
        self.input_data_path = "./expert_eval_for_selected_hyp_in_exp_{}.json".format(exp_id)
        # self.name: "Wanhao" or "Ben" or "Penghui"
        self.name = None
        # self.data: {bkg_id: {q_id: [gene_hyp, gdth_hyp, cnt_matched_insp, cur_matched_score, cur_matched_score_reason, expert_matched_score]}}
        self.data = self.load_data()
        self.num_q_per_bkg = 4
        # start of the second expert
        self.seperate_bkg_id = 30

    def load_data(self):
        with open(self.input_data_path, "r") as f:
            data = json.load(f)
        return data


    def save_data(self, data):
        with open(self.output_data_path, "w") as f:
            json.dump(data, f)


    def start_eval(self):
        name = None
        while (name != "Wanhao" and name != "Ben" and name != "Penghui"):
            name = input("Please input your name: (please input Wanhao or Ben or Penghui): ")
        self.name = name
        self.output_data_path = "./expert_eval_for_selected_hyp_in_exp_{}_{}.json".format(self.exp_id, self.name)

        if name == "Wanhao":
            id_bkg_list = [str(i) for i in range(0, self.seperate_bkg_id)]
        elif name == "Ben":
            id_bkg_list = [str(i) for i in range(self.seperate_bkg_id, 51)]
        elif name == "Penghui":
            id_bkg_list = [str(i) for i in range(0, 6)] + [str(i) for i in range(self.seperate_bkg_id, self.seperate_bkg_id + 6)]
        else:
            raise ValueError("Invalid name")
        
        # recover the previous output data
        last_bkg_id_to_resume = None
        last_q_id_to_resume = None
        if os.path.exists(self.output_data_path):
            with open(self.output_data_path, "r") as f:
                prev_output_file = json.load(f)
                self.data = prev_output_file
                # print("prev_output_file.keys(): ", prev_output_file.keys())
            # get the lastest background id and question id to resume (last_bkg_id_to_resume and last_q_id_to_resume)
            if_already_identified_ids = False
            # for bkg_id in prev_output_file:
            for bkg_id in id_bkg_list:
                # print("bkg_id: ", bkg_id)
                for q_id in range(len(prev_output_file[bkg_id])):
                    # print("q_id: ", q_id)
                    if len(prev_output_file[bkg_id][q_id]) == 6:
                        last_bkg_id_to_resume = bkg_id
                        last_q_id_to_resume = q_id
                    elif len(prev_output_file[bkg_id][q_id]) == 5:
                        if last_bkg_id_to_resume != None and last_q_id_to_resume != None:
                            last_bkg_id_to_resume = bkg_id
                            last_q_id_to_resume = q_id
                        if_already_identified_ids = True
                        break
                    else:
                        raise ValueError("Invalid length of data: {}".format(len(prev_output_file[bkg_id][q_id])))
                if if_already_identified_ids:
                    break
            if last_bkg_id_to_resume != None or last_q_id_to_resume != None:
                assert last_bkg_id_to_resume != None and last_q_id_to_resume != None
            if if_already_identified_ids == False:
                return print("The previous output file should have been finished.")
            print("Resume from the last background ID: {}, question ID: {}".format(last_bkg_id_to_resume, last_q_id_to_resume))
        else:
            print("There are in total {} questions to evaluate".format(len(id_bkg_list)*self.num_q_per_bkg))
        time.sleep(3)


        output_data_with_expert_eval = copy.deepcopy(self.data)
        for cur_bkg_id in id_bkg_list:
            # resume from the last bkg_id
            if last_bkg_id_to_resume != None:
                if int(cur_bkg_id) < int(last_bkg_id_to_resume):
                    continue
            len_data_cur_bkg_id = len(self.data[cur_bkg_id])
            assert len_data_cur_bkg_id == self.num_q_per_bkg
            for cur_d_id in range(len_data_cur_bkg_id):
                # resume from the last q_id
                if last_q_id_to_resume != None:
                    if cur_bkg_id == last_bkg_id_to_resume and cur_d_id < last_q_id_to_resume:
                        continue
                # cur_d: [gene_hyp, gdth_hyp, cnt_matched_insp, cur_matched_score, cur_matched_score_reason]
                cur_d = self.data[cur_bkg_id][cur_d_id]
                print("\n\nBackground ID: {}, Question ID: {}".format(cur_bkg_id, cur_d_id))
                print("\nGenerated Hypothesis: \n{}".format(cur_d[0]))
                print("\nGroundtruth Hypothesis: \n{}".format(cur_d[1]))
                print("\nAutomatic matched score from GPT4o: {}".format(cur_d[3]))
                print("\nReason for the matched score: \n{}".format(cur_d[4]))
                print("\nDo you think the automatic matched score is correct? (y/n)")
                is_reasonable = None
                while (is_reasonable != "y" and is_reasonable != "n"):
                    is_reasonable = input()
                if is_reasonable == "n":
                    print("Please input your matched score (0-5)")
                    expert_matched_score = None
                    while (expert_matched_score is None or expert_matched_score < 0 or expert_matched_score > 5):
                        expert_matched_score = input()
                        try:
                            expert_matched_score = int(expert_matched_score)
                        except:
                            print("Please input an integer between 0 and 5")
                            expert_matched_score = None
                elif is_reasonable == "y":
                    expert_matched_score = cur_d[3]
                else:
                    raise ValueError("Invalid input")
                output_data_with_expert_eval[cur_bkg_id][cur_d_id].append(expert_matched_score)
                # save data for every question
                self.save_data(output_data_with_expert_eval)
        print("All questions have been evaluated. Thank you for your time!")
                
                    














if __name__ == "__main__":
    expert_eval = ExpertEval(exp_id=8)
    expert_eval.start_eval()
