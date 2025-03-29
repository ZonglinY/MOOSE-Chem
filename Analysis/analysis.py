from ast import List
import numpy as np
import json, random, copy, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_dict_title_2_abstract, recover_generated_title_to_exact_version_of_title, load_bkg_and_insp_from_chem_annotation, load_chem_annotation, if_element_in_list_with_similarity_threshold
from sympy import N
np.set_printoptions(precision=2)



## Input:
# list_of_list_of_scores: [[score0, score1, score2, score3], [score0, score1, score2, score3], ...]
## Output:
# ave_list_of_scores: [ave_score0, ave_score1, ave_score2, ave_score3]
# max_list_of_scores: [max_score0, max_score1, max_score2, max_score3]
# best_ave_list_of_scores_vertical: [best_ave_score0, best_ave_score1, best_ave_score2, best_ave_score3]
def analysis_list_of_list_of_scores(list_of_list_of_scores):
    list_of_list_of_scores = np.array(list_of_list_of_scores, dtype=np.float32)
    ave_list_of_scores = np.mean(list_of_list_of_scores, axis=0)
    max_list_of_scores = np.max(list_of_list_of_scores, axis=0)
    ave_list_of_scores_vertical = np.mean(list_of_list_of_scores, axis=1)
    best_ave_list_of_scores_vertical = -np.sort(-ave_list_of_scores_vertical, axis=0)
    return ave_list_of_scores, max_list_of_scores, best_ave_list_of_scores_vertical


## Input:
# final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
#       hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}
# target_bkg: text
# gold_insps: [gold_insp0, gold_insp1, ...] (list of text)
## Output:
# gold_insp_scores, other_insp_scores: [ave_score0, ave_score1, ave_score2, ave_score3]
def compare_score_between_gold_insp_and_others(final_data_collection_path, target_bkg, gold_insps):
    # load final_data_collection
    with open(final_data_collection_path, 'r') as f:
        final_data_collection = json.load(f)

    # check input parameters
    assert len(final_data_collection) == 1
    assert len(gold_insps) >= 1
    # get rid of the leading and trailing whitespaces in target_bkg
    target_bkg = target_bkg.strip()
    # get rid of the leading and trailing whitespaces in gold_insps
    for cur_insp_id, cur_insp in enumerate(gold_insps):
        gold_insps[cur_insp_id] = cur_insp.strip()
    assert target_bkg in final_data_collection, print("target_bkg: {}; final_data_collection: {}".format(target_bkg, final_data_collection.keys()))

    # looping to find eval scores
    gold_insp_scores, other_insp_scores = [], []
    for cur_insp_title, cur_insp_hypothesis in final_data_collection[target_bkg].items():
        cur_insp_title = cur_insp_title.strip()
        print("cur_insp_title: ", cur_insp_title)
        for cur_mutation_id, cur_mutation_hypothesis in cur_insp_hypothesis.items():
            if "inter_recom" not in cur_mutation_id and "self_explore" not in cur_mutation_id:
                # cur_mutation_hypothesis: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ..., [hypn, reasoning processn, feedbackn, [scoren, score_reasonn]]]
                cur_score = cur_mutation_hypothesis[-1][-1][0]
                if cur_insp_title in gold_insps:
                    gold_insp_scores.append(cur_score)
                else:
                    other_insp_scores.append(cur_score)

    # summarize results
    print("length of gold_insp_scores: {}; length of other_insp_scores: {}".format(len(gold_insp_scores), len(other_insp_scores)))
    ave_gold_insp_scores, max_gold_insp_scores, best_ave_gold_insp_score_vertical = analysis_list_of_list_of_scores(gold_insp_scores)
    if_have_other_insp_scores = False
    if len(other_insp_scores) > 0:
        if_have_other_insp_scores = True
        other_insp_scores = np.array(other_insp_scores, dtype=np.float32)
        ave_other_insp_scores = np.mean(other_insp_scores, axis=0)
        max_other_insp_scores = np.max(other_insp_scores, axis=0)
    
    print("\nave_gold_insp_scores: {}; max_gold_insp_scores: {}".format(ave_gold_insp_scores, max_gold_insp_scores))        
    if if_have_other_insp_scores:
        print("ave_othr_insp_scores: {}; max_othr_insp_scores: {}".format(ave_other_insp_scores, max_other_insp_scores))
        return ave_gold_insp_scores, ave_other_insp_scores, max_gold_insp_scores, max_other_insp_scores
    else:
        return ave_gold_insp_scores, None, max_gold_insp_scores, None



## Function
# compare_score_between_inter_recom_and_self_explore
def compare_score_between_inter_recom_and_self_explore(final_data_collection_path, target_bkg):
    # load final_data_collection
    with open(final_data_collection_path, 'r') as f:
        final_data_collection = json.load(f)

    # check input parameters: should only have one background question as key (the current code is only designed for one background question)
    assert len(final_data_collection) == 1
    # get rid of the leading and trailing whitespaces in target_bkg
    target_bkg = target_bkg.strip()

    # looping to find eval scores
    inter_recom_scores, self_explore_scores = [], []
    # selected for inter_recom and self_explore
    original_scores_selected_inter_recom, original_scores_selected_self_explore = [], []
    for cur_insp_title, cur_insp_hypothesis in final_data_collection[target_bkg].items():
        cur_insp_title = cur_insp_title.strip()
        for cur_mutation_id, cur_mutation_hypothesis in cur_insp_hypothesis.items():
            # cur_insp_hypothesis['inter_com']: {matched_insp_title0: [[hyp0, reasoning process0, feedback0], ...], ...}
            if "inter_recom" in cur_mutation_id:
                # print("len(cur_mutation_hypothesis.keys()): ", len(cur_mutation_hypothesis.keys()))
                # cur_best_previous_mutation_id: the mutation_id during the first round that is selected for inter_recom (usually the highest score one)
                for cur_best_previous_mutation_id, cur_mutation_hypothesis_under_prev_mut_id in cur_mutation_hypothesis.items():
                    for matched_insp_title, matched_hypthesis in cur_mutation_hypothesis_under_prev_mut_id.items():
                        cur_mutation_score = matched_hypthesis[-1][-1][0]
                        # print("cur_mutation_score: ", cur_mutation_score)
                        inter_recom_scores.append(cur_mutation_score)
                    # original_scors_selected
                    original_scores_selected_inter_recom.append(cur_insp_hypothesis[cur_best_previous_mutation_id][-1][-1][0])
            # cur_insp_hypothesis['self_explore']: {selected_best_mutation_id: {mutation_id: [[extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0], ...]}}
            elif "self_explore" in cur_mutation_id:
                # print("len(cur_mutation_hypothesis.keys()): ", len(cur_mutation_hypothesis.keys()))
                # cur_best_previous_mutation_id: the mutation_id during the first round that is selected for inter_recom (usually the highest score one)
                for cur_best_previous_mutation_id, cur_mutation_hypothesis_under_prev_mut_id in cur_mutation_hypothesis.items():
                    for self_mutate_id, matched_hypthesis in cur_mutation_hypothesis_under_prev_mut_id.items():
                        cur_explore_score = matched_hypthesis[-1][-1][0]
                        # print("cur_explore_score: ", cur_explore_score)
                        self_explore_scores.append(cur_explore_score)
                    # original_scors_selected
                    original_scores_selected_self_explore.append(cur_insp_hypothesis[cur_best_previous_mutation_id][-1][-1][0])
    
    # summarize results
    print("\n\nlength of inter_recom_scores: {}; length of self_explore_scores: {}".format(len(inter_recom_scores), len(self_explore_scores)))

    # inter_recom
    ave_inter_recom_scores, max_list_of_scores, best_ave_inter_recom_score_vertical = analysis_list_of_list_of_scores(inter_recom_scores)
    # self_explore
    ave_self_explore_scores, max_self_explore_scores, best_ave_self_explore_score_vertical = analysis_list_of_list_of_scores(self_explore_scores)
    # original_scores_selected before the second round of inspiration selection or self_explore
    ave_original_scores_selected_inter_recom, max_original_scores_selected_inter_recom, best_ave_original_scores_selected_inter_recom = analysis_list_of_list_of_scores(original_scores_selected_inter_recom)
    ave_original_scores_selected_self_explore, max_original_scores_selected_self_explore, best_ave_original_scores_selected_self_explore = analysis_list_of_list_of_scores(original_scores_selected_self_explore)

    print("\nave_original_scores_selected_self_explore: {}; \nave_original_scores_selected_inter_recom: {}; \nave_self_explore_scores: {}; \nave_inter_recom_scores: {}".format(ave_original_scores_selected_self_explore, ave_original_scores_selected_inter_recom, ave_self_explore_scores, ave_inter_recom_scores))
    print("\nbest_ave_original_scores_selected_self_explore: {}; \nbest_ave_original_scores_selected_inter_recom: {}; \nbest_ave_self_explore_score_vertical: {}; \nbest_ave_inter_recom_score_vertical: {}".format(best_ave_original_scores_selected_self_explore, best_ave_original_scores_selected_inter_recom, best_ave_self_explore_score_vertical, best_ave_inter_recom_score_vertical))
    return ave_inter_recom_scores, ave_self_explore_scores, ave_original_scores_selected_inter_recom, ave_original_scores_selected_self_explore
                



def find_highest_scored_hypothesis_from_first_round(final_data_collection_path, target_bkg, display_rank_idx=0):
    # load final_data_collection
    with open(final_data_collection_path, 'r') as f:
        final_data_collection = json.load(f)

    best_ave_score_list = []
    best_hypothesis_list, best_score_list = [], []
    best_first_round_mode_list, best_first_round_insp_list = [], []
    for cur_insp_title, cur_insp_hypothesis in final_data_collection[target_bkg].items():
        for cur_mutation_id, cur_mutation_hypothesis in cur_insp_hypothesis.items():
            if "inter_recom" not in cur_mutation_id and "self_explore" not in cur_mutation_id:
                cur_mutation_score = cur_mutation_hypothesis[-1][-1][0]
                cur_mutation_ave_score = np.mean(cur_mutation_score[:])
                # find index in lists to insert
                if len(best_ave_score_list) == 0:
                    cur_shared_index = 0
                else:
                    cur_shared_index = 0
                    for cur_best_ave_score in best_ave_score_list:
                        if cur_mutation_ave_score > cur_best_ave_score:
                            break
                        cur_shared_index += 1
                # insert
                best_ave_score_list.insert(cur_shared_index, cur_mutation_ave_score)
                best_hypothesis_list.insert(cur_shared_index, cur_mutation_hypothesis[-1][0])
                best_score_list.insert(cur_shared_index, cur_mutation_score)
                best_first_round_mode_list.insert(cur_shared_index, cur_mutation_id)
                best_first_round_insp_list.insert(cur_shared_index, cur_insp_title)
    
    print("\nrank: {}; score: {}; best_first_round_mode: {}; \nbest_first_round_insp: {}; \nbest_hypothesis: {}".format(display_rank_idx, best_score_list[display_rank_idx], best_first_round_mode_list[display_rank_idx], best_first_round_insp_list[display_rank_idx], best_hypothesis_list[display_rank_idx]))
    



def find_highest_scored_hypothesis_from_second_round(final_data_collection_path, target_bkg, display_rank_idx=0, round_id=2):
    # load final_data_collection
    with open(final_data_collection_path, 'r') as f:
        final_data_collection = json.load(f)

    inter_recom_mut_id = "inter_recom_{}".format(round_id-1)
    self_explore_mut_id = "self_explore_{}".format(round_id-1)

    best_ave_score_list = []
    best_hypothesis_list, best_score_list = [], []
    best_second_round_mode_list, best_first_round_insp_list, best_second_round_insp_list = [], [], []
    for cur_insp_title, cur_insp_hypothesis in final_data_collection[target_bkg].items():
        for cur_mutation_id, cur_mutation_hypothesis in cur_insp_hypothesis.items():
            if cur_mutation_id == inter_recom_mut_id:
                for cur_best_previous_mutation_id, cur_mutation_hypothesis_under_prev_mut_id in cur_mutation_hypothesis.items():
                    for matched_insp_title, matched_hypthesis in cur_mutation_hypothesis_under_prev_mut_id.items():
                        cur_mutation_score = matched_hypthesis[-1][-1][0]
                        cur_mutation_ave_score = np.mean(cur_mutation_score[:])
                        # find index in lists to insert
                        if len(best_ave_score_list) == 0:
                            cur_shared_index = 0
                        else:
                            cur_shared_index = 0
                            for cur_best_ave_score in best_ave_score_list:
                                if cur_mutation_ave_score > cur_best_ave_score:
                                    break
                                cur_shared_index += 1
                        # insert 
                        best_ave_score_list.insert(cur_shared_index, cur_mutation_ave_score)
                        best_hypothesis_list.insert(cur_shared_index, matched_hypthesis[-1][0])
                        best_score_list.insert(cur_shared_index, cur_mutation_score)
                        best_second_round_mode_list.insert(cur_shared_index, inter_recom_mut_id)
                        best_first_round_insp_list.insert(cur_shared_index, cur_insp_title)
                        best_second_round_insp_list.insert(cur_shared_index, matched_insp_title)
            elif cur_mutation_id == self_explore_mut_id:
                for cur_best_previous_mutation_id, cur_mutation_hypothesis_under_prev_mut_id in cur_mutation_hypothesis.items():
                    for self_mutate_id, matched_hypthesis in cur_mutation_hypothesis_under_prev_mut_id.items():
                        cur_explore_score = matched_hypthesis[-1][-1][0]
                        cur_explore_ave_score = np.mean(cur_explore_score[:])
                        # find index in lists to insert
                        if len(best_ave_score_list) == 0:
                            cur_shared_index = 0
                        else:
                            cur_shared_index = 0
                            for cur_best_ave_score in best_ave_score_list:
                                if cur_explore_ave_score > cur_best_ave_score:
                                    break
                                cur_shared_index += 1
                        # insert
                        best_ave_score_list.insert(cur_shared_index, cur_explore_ave_score)
                        best_hypothesis_list.insert(cur_shared_index, matched_hypthesis[-1][0])
                        best_score_list.insert(cur_shared_index, cur_explore_score)
                        best_second_round_mode_list.insert(cur_shared_index, "self_explore")
                        best_first_round_insp_list.insert(cur_shared_index, cur_insp_title)
                        best_second_round_insp_list.insert(cur_shared_index, self_mutate_id)

    print("\nrank: {}; score: {}; best_second_round_mode: {}; \nbest_first_round_insp: {}; \nbest_second_round_insp: {}; \nbest_hypothesis: {}".format(display_rank_idx, best_score_list[display_rank_idx], best_second_round_mode_list[display_rank_idx], best_first_round_insp_list[display_rank_idx], best_second_round_insp_list[display_rank_idx], best_hypothesis_list[display_rank_idx]))


def compare_similarity_between_inspiration_retrieval_and_similarity_retrieval(insp_file_path, simi_file_path, title_abstract_all_insp_literature_path="./title_abstract.json"):
    # dict_title_2_abstract: {'title': 'abstract', ...}
    title_abstract_collector, dict_title_2_abstract = load_dict_title_2_abstract(title_abstract_collector_path=title_abstract_all_insp_literature_path)     
    groundtruth_insp_titles = list(dict_title_2_abstract.keys())

    with open(insp_file_path, 'r') as f:
        insp_data = json.load(f)
        insp_data = insp_data[0]
    with open(simi_file_path, 'r') as f:
        simi_data = json.load(f)
        simi_data = simi_data[0]

    # get bkg_q
    assert insp_data.keys() == simi_data.keys()
    assert len(insp_data.keys()) == 1
    bkg_q = list(insp_data.keys())[0]

    insp_matched_titles = [recover_generated_title_to_exact_version_of_title(groundtruth_insp_titles, insp_title[0]) for insp_title in insp_data[bkg_q]]
    simi_matched_titles = [recover_generated_title_to_exact_version_of_title(groundtruth_insp_titles, simi_title[0]) for simi_title in simi_data[bkg_q]]

    same_titles, diff_titles = [], []
    for cur_t in insp_matched_titles:
        if cur_t in simi_matched_titles:
            same_titles.append(cur_t)
        else:
            diff_titles.append(cur_t)
    same_ratio = len(same_titles) / len(insp_matched_titles)
    print("\n\nsame_titles: {}; diff_titles: {}; same_ratio: {:.2f}".format(len(same_titles), len(diff_titles), same_ratio))



def check_moosechem_output():
    # display_rank_idx: the rank of the hypothesis (based on its average score) to display
    chem_annotation_path = "./Data/chem_research_2024.xlsx"
    background_question_id = 36
    display_rank_idx = 2
    if_use_strict_survey_question = 1

    final_data_collection_path = "./Checkpoints/hypothesis_generation_gpt4_selfea_1_interea_1_bkgid_{}.json".format(background_question_id)

    target_bkg, gold_insps = load_bkg_and_insp_from_chem_annotation(chem_annotation_path, background_question_id, if_use_strict_survey_question)
    print("len(gold_insps): ", len(gold_insps))

    ave_gold_insp_scores, ave_other_insp_scores, max_gold_insp_scores, max_other_insp_scores = compare_score_between_gold_insp_and_others(final_data_collection_path, target_bkg, gold_insps)

    # ave_inter_recom_scores, ave_self_explore_scores, max_inter_recom_scores, max_self_explore_scores = compare_score_between_inter_recom_and_self_explore(final_data_collection_path, target_bkg)

    find_highest_scored_hypothesis_from_first_round(final_data_collection_path, target_bkg, display_rank_idx=display_rank_idx)

    find_highest_scored_hypothesis_from_second_round(final_data_collection_path, target_bkg, display_rank_idx=display_rank_idx)



def check_difference_inspiration_retrieval_similarity_retrieval():
    bkgid = 17
    num_screen = 15
    if_with_survey = 1

    print("bkgid: {}; num_screen: {}; if_with_survey: {}".format(bkgid, num_screen, if_with_survey))
    insp_file_path = "./Checkpoints/coarse_inspiration_search_gpt4_numScreen_{}_limited_survey_bkgid_{}_similarity_0_ifSurvey_{}.json".format(num_screen, bkgid, if_with_survey)
    simi_file_path = "./Checkpoints/coarse_inspiration_search_gpt4_numScreen_{}_limited_survey_bkgid_{}_similarity_1_ifSurvey_{}.json".format(num_screen, bkgid, if_with_survey)
    compare_similarity_between_inspiration_retrieval_and_similarity_retrieval(insp_file_path=insp_file_path, simi_file_path=simi_file_path)



## Function
# obtain average groundtruth inspirations top3 hit ratio during screening; used for Assumption 1
## Input
# file_root_name_path: a path, with id.json as ending, e.g., "./Checkpoints/coarse_inspiration_search_gpt4_numScreen__15_similarity_0_round_3_bkgid_"
# data_id_range: [start_id, end_id], including both start_id and end_id
# round_id: an integer, indicating which round of screened insp to calculate averaged hit ratio; if it is -1, then calculate the averaged hit ratio for all rounds
## Output
# ave_hit_ratio_top3: a float number
def get_average_screened_insp_hit_ratio_from_a_series_of_files(file_root_name_path, data_id_range, round_id):
    assert len(data_id_range) == 2
    assert data_id_range[1] >= data_id_range[0]
    hit_ratio_top3_collection = []
    for cur_id in range(data_id_range[0], data_id_range[1]+1):
        # print("cur_id: ", cur_id)
        cur_file_path = file_root_name_path + str(cur_id) + ".json"
        if not os.path.exists(cur_file_path):
            print("Warning: file not exists: ", cur_file_path)
            continue
        with open(cur_file_path, 'r') as f:
            cur_data = json.load(f)
        cur_hit_ratio_data = cur_data[1]
        cur_bkg_key = list(cur_hit_ratio_data.keys())
        if len(cur_bkg_key) > 1:
            print("Warning: one file contains more than one background question: ", cur_bkg_key)
        for tmp_bkg_key in cur_bkg_key:
            for cur_round_id in range(len(cur_hit_ratio_data[tmp_bkg_key])):
                if round_id == -1 or cur_round_id == round_id:
                    cur_hit_ratio_numbers = cur_hit_ratio_data[tmp_bkg_key][cur_round_id]
                    cur_hit_ratio_top3_number = cur_hit_ratio_numbers[1]
                    hit_ratio_top3_collection.append(cur_hit_ratio_top3_number)
    ave_hit_ratio_top3 = np.mean(hit_ratio_top3_collection)
    print("round: {}; cnt_data_averaged: {}; ave_hit_ratio_top3: {:.3f}".format(round_id, len(hit_ratio_top3_collection), ave_hit_ratio_top3))
    return ave_hit_ratio_top3



## Function
# obtain the best matched score for each background; used for Assumption 2
#   this function analyze all hypotheses with a Matched Score (or cnt_matched_insp >= 1), while overlook all hypotheses without a Matched Score (or cnt_matched_insp == 0)
## Input
# file_root_name_path: a path, with id.json as ending, e.g., "./Checkpoints/evaluation_gpt4_intraEA_1_interEA_1_gdthInsp_1_bkgid_"
# data_id_range: [start_id, end_id], including both start_id and end_id
# get_expert_eval_file_type: 0: don't collect hypotheses collection file for expert evaluation; 1: collect 2 top and 2 random (for exp5); 2: collect 4 top (for exp8)
# if_save: if save the hypotheses collection file for expert evaluation
# if_not_only_from_gdth_insp: if the file in file_root_name_path is not only from gdth hypotheses (if not, cnt_matched_insp might be zero, and therefore cur_matched_insp_hyp_collection file could be None or empty)
def get_top_matched_score_for_each_background(file_root_name_path, data_id_range, chem_annotation_path="./Data/chem_research_2024.xlsx", if_use_strict_survey_question=1, get_expert_eval_file_type=0, if_save=False, if_not_only_from_gdth_insp=False):
    assert len(data_id_range) == 2
    assert data_id_range[1] >= data_id_range[0]
    assert get_expert_eval_file_type in [0, 1, 2]
    bkg_q_list, dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2idx, dict_idx2bkg, dict_bkg2reasoningprocess = load_chem_annotation(chem_annotation_path, if_use_strict_survey_question)  
    # top_matched_score_collection: {cnt_matched_insp: [top_matched_score0, top_matched_score1, ...], ...}
    top_matched_score_collection = {}
    # top_matched_score_collection_cnt_matched_insp: {cnt_matched_insp: {top_matched_score0: cnt0, top_matched_score1: cnt1, ...}, ...}
    top_matched_score_collection_cnt_matched_insp = {}
    # ave_matched_score_collection: {cnt_matched_insp: [ave_matched_score0, ave_matched_score1, ...], ...}
    ave_matched_score_collection = {}
    # ave_matched_score_collection_cnt_matched_insp: {cnt_matched_insp: {ave_matched_score0: cnt0, ave_matched_score1: cnt1, ...}, ...}
    ave_matched_score_collection_cnt_matched_insp = {}
    # selected_hyp_for_expert_eval: {bkg_id: [[cur_hyp0 (top 1), cur_gdth_hyp0, cnt_matched_insp0, cur_matched_score0, cur_matched_score_reason0], [hyp1 (top 2), ...], [hyp2 (random 1), ...], [hyp3 (random 2), ...]], ...}
    if get_expert_eval_file_type != 0:
        selected_hyp_for_expert_eval = {}
    for cur_id in range(data_id_range[0], data_id_range[1]+1):
        # cur_gdth_insp_necessary_cnt
        cur_bkg_q = bkg_q_list[cur_id]
        cur_reasoning_process = dict_bkg2reasoningprocess[cur_bkg_q]
        cur_gdth_insp_necessary_cnt = cur_reasoning_process.count("+")
        assert cur_gdth_insp_necessary_cnt in [1, 2, 3]
        # load file
        cur_file_path = file_root_name_path + str(cur_id) + ".json"
        with open(cur_file_path, 'r') as f:
            cur_data = json.load(f)
        # cur_matched_score_collection: [top_matched_score0, top_matched_score1, ...] (sorted in descending order)
        cur_matched_score_collection = []
        # cur_matched_insp_hyp_collection: [[cur_hyp, cur_gdth_hyp, cur_ave_score, cur_scores, cnt_matched_insp, cur_used_insps_set, cur_full_gdth_insps, cur_matched_score, cur_matched_score_reason, cur_round_id], ...]; ranked with cnt_matched_insp
        cur_matched_insp_hyp_collection = cur_data[2]
        # rank cur_matched_insp_hyp_collection by cur_matched_score, in descending order
        cur_matched_insp_hyp_collection = sorted(cur_matched_insp_hyp_collection, key=lambda x: int(x[7]), reverse=True)
        if len(cur_matched_insp_hyp_collection) >= 25:
            print("Warning: too many generated hypotheses for experiments using groundtruth inspiration: {}".format(len(cur_matched_insp_hyp_collection)))
        for cur_hyp_info_id in range(len(cur_matched_insp_hyp_collection)):
            # cur_found_insps
            cur_used_mutation_ids = cur_matched_insp_hyp_collection[cur_hyp_info_id][5]
            # print("cur_used_mutation_ids: ", cur_used_mutation_ids)
            cur_found_insps = []
            for cur_mut_id in cur_used_mutation_ids:
                if ";" in cur_mut_id:
                    cur_found_insps += cur_mut_id.split(";")
                else:
                    cur_found_insps.append(cur_mut_id)
            # if we suspect intra-EA might lead to not better hypothesis in line '1' and line '2', we can add the following lines
            # if '1' in cur_found_insps or '2' in cur_found_insps:
            #     continue
            cur_matched_score = cur_matched_insp_hyp_collection[cur_hyp_info_id][7]
            cur_matched_score = int(cur_matched_score)
            cur_matched_score_collection.append(cur_matched_score)
        cur_matched_score_collection = sorted(cur_matched_score_collection, reverse=True)
        # top_matched_score, ave_matched_score
        if len(cur_matched_score_collection) == 0:
            assert if_not_only_from_gdth_insp == True
            top_matched_score, ave_matched_score = 0, 0
        else:
            top_matched_score = cur_matched_score_collection[0]
            ave_matched_score = np.mean(cur_matched_score_collection)
            
        # top_matched_score_collection
        if cur_gdth_insp_necessary_cnt not in top_matched_score_collection:
            top_matched_score_collection[cur_gdth_insp_necessary_cnt] = []  
        top_matched_score_collection[cur_gdth_insp_necessary_cnt].append(top_matched_score)
        # top_matched_score_collection_cnt_matched_insp
        if cur_gdth_insp_necessary_cnt not in top_matched_score_collection_cnt_matched_insp:
            top_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt] = {}
        if top_matched_score not in top_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt]:
            top_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt][top_matched_score] = 0
        top_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt][top_matched_score] += 1
        # ave_matched_score_collection
        if cur_gdth_insp_necessary_cnt not in ave_matched_score_collection:
            ave_matched_score_collection[cur_gdth_insp_necessary_cnt] = []
        ave_matched_score_collection[cur_gdth_insp_necessary_cnt].append(ave_matched_score)
        # ave_matched_score_collection_cnt_matched_insp
        if cur_gdth_insp_necessary_cnt not in ave_matched_score_collection_cnt_matched_insp:
            ave_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt] = {}
        # round_matched_score: x.4 --> x; x.5 --> x + 1
        ave_matched_score_decimal_part = ave_matched_score % 1
        round_matched_score = int(ave_matched_score) + 1 if ave_matched_score_decimal_part >= 0.5 else int(ave_matched_score)
        if round_matched_score not in ave_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt]:
            ave_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt][round_matched_score] = 0
        ave_matched_score_collection_cnt_matched_insp[cur_gdth_insp_necessary_cnt][round_matched_score] += 1
        # get_expert_eval_file_type
        if get_expert_eval_file_type == 1:
            # selected_hyp_for_expert_eval (top 2 + random 2)
            selected_hyp_for_expert_eval[cur_id] = []
            assert len(cur_matched_insp_hyp_collection) >= 4, print("len(cur_matched_insp_hyp_collection): ", len(cur_matched_insp_hyp_collection))
            random_ids = np.random.choice(range(2, len(cur_matched_insp_hyp_collection)), 2, replace=False)
            selected_ids = [0, 1] + list(random_ids)
            for cur_selected_id in selected_ids:
                selected_hyp_for_expert_eval[cur_id].append([cur_matched_insp_hyp_collection[cur_selected_id][0], cur_matched_insp_hyp_collection[cur_selected_id][1], cur_matched_insp_hyp_collection[cur_selected_id][4], cur_matched_insp_hyp_collection[cur_selected_id][7], cur_matched_insp_hyp_collection[cur_selected_id][8]])
        elif get_expert_eval_file_type == 2:
            # selected_hyp_for_expert_eval (top 4)
            selected_hyp_for_expert_eval[cur_id] = []
            collect_size = 4
            if len(cur_matched_insp_hyp_collection) < collect_size:
                print("Warning: len(cur_matched_insp_hyp_collection): ", len(cur_matched_insp_hyp_collection))
                collect_size = len(cur_matched_insp_hyp_collection)
            for cur_selected_id in range(collect_size):
                selected_hyp_for_expert_eval[cur_id].append([cur_matched_insp_hyp_collection[cur_selected_id][0], cur_matched_insp_hyp_collection[cur_selected_id][1], cur_matched_insp_hyp_collection[cur_selected_id][4], cur_matched_insp_hyp_collection[cur_selected_id][7], cur_matched_insp_hyp_collection[cur_selected_id][8]])


    print("top_matched_score_collection_cnt_matched_insp: ", top_matched_score_collection_cnt_matched_insp)
    print("ave_matched_score_collection_cnt_matched_insp: ", ave_matched_score_collection_cnt_matched_insp)

    print("\ntop_matched_score_collection: ", top_matched_score_collection)
    # ave_top_matched_score_collection
    ave_top_matched_score_collection = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in top_matched_score_collection.items()}
    # len_top_matched_score_collection
    len_top_matched_score_collection = {tmp_key: len(tmp_value) for tmp_key, tmp_value in top_matched_score_collection.items()}
    # ave_top_matched_score
    ave_top_matched_score = np.mean([tmp_value for tmp_value in ave_top_matched_score_collection.values()])
    # len_top_matched_score
    len_top_matched_score = np.sum([tmp_value for tmp_value in len_top_matched_score_collection.values()])
    print("len_top_matched_score_collection: {}".format(len_top_matched_score_collection))
    print("ave_top_matched_score_collection: ", {k:round(v,3) for k,v in ave_top_matched_score_collection.items()})
    print("\nlen_top_matched_score: {}; ave_top_matched_score: {:.3f}".format(len_top_matched_score, ave_top_matched_score))

    print("\n\nave_matched_score_collection: ", ave_matched_score_collection)
    # ave_ave_matched_score_collection
    ave_ave_matched_score_collection = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in ave_matched_score_collection.items()}
    # len_ave_matched_score_collection
    len_ave_matched_score_collection = {tmp_key: len(tmp_value) for tmp_key, tmp_value in ave_matched_score_collection.items()}
    # len_ave_matched_score
    len_ave_matched_score = np.sum([tmp_value for tmp_value in len_ave_matched_score_collection.values()])
    # ave_ave_matched_score
    ave_ave_matched_score = 0
    for tmp_key, tmp_value in ave_matched_score_collection.items():
        ave_ave_matched_score += len(tmp_value) * ave_ave_matched_score_collection[tmp_key]
    ave_ave_matched_score /= len_ave_matched_score
    print("len_ave_matched_score_collection: {}".format(len_ave_matched_score_collection))
    print("ave_ave_matched_score_collection: ", {k:round(v,3) for k,v in ave_ave_matched_score_collection.items()})
    print("\nlen_ave_matched_score: {}; ave_ave_matched_score: {:.3f}".format(len_ave_matched_score, ave_ave_matched_score))

    # save selected_hyp_for_expert_eval
    if if_save:
        if get_expert_eval_file_type == 1:
            with open("./expert_eval_for_selected_hyp_in_exp_5.json", 'w') as f:
                json.dump(selected_hyp_for_expert_eval, f)
        elif get_expert_eval_file_type == 2:
            with open("./expert_eval_for_selected_hyp_in_exp_8.json", 'w') as f:
                json.dump(selected_hyp_for_expert_eval, f)

          



# to see how many components are shared between two lists (based on Jaccard similarity)
def count_intersection_with_jaccard_similarity(gene_list, gold_list):
    cnt_intersection = 0
    for cur_gold in gold_list:
        # if cur_gold in gene_list:
        if if_element_in_list_with_similarity_threshold(gene_list, cur_gold, threshold=0.65):
            cnt_intersection += 1
    return cnt_intersection


# found_insps： [title0, title1, "0", "inter_recom", "self_explore", ...]
# return: [title0, title1, ...]
def get_rid_of_mutation_ids_in_found_insps(found_insps):
    new_found_insps = []
    for cur_insps in found_insps:
        cur_insps = cur_insps.strip()
        if cur_insps.isdigit():
            continue
        if cur_insps == "recom":
            continue
        if "inter_recom" in cur_insps and cur_insps[-1].isdigit():
            continue
        if "self_explore" in cur_insps and cur_insps[-1].isdigit():
            continue
        new_found_insps.append(cur_insps)
    assert len(new_found_insps) <= 3, print("new_found_insps: ", new_found_insps)
    assert len(new_found_insps) >= 1
    return new_found_insps




## Input
# file_root_name_path: a path, with id.json as ending, e.g., "./Checkpoints/evaluation_gpt4_intraEA_1_interEA_1_gdthInsp_1_bkgid_"
# data_id_range: [start_id, end_id], including both start_id and end_id
# max_step: -1: include all steps; if positive: only include steps <= max_step
# Q: currently if the ranking is [4,4,4,4], their rank id is [1,2,3,4]; but it might be better to be [2.5,2.5,2.5,2.5]
def get_average_ranking_position_for_hyp_with_gdth_insp(file_root_name_path, data_id_range, chem_annotation_path="./Data/chem_research_2024.xlsx", if_random_order=False, keep_top_ratio=1.0, max_step=-1):
    assert len(data_id_range) == 2
    assert data_id_range[1] >= data_id_range[0]
    assert if_random_order in [True, False]
    assert max_step in [-1, 0, 1, 2, 3]

    # list_hyp_info: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id], matched_score, matched_score_reason], ...]
    def check_whether_an_ranked_hypothesis_collection_item_in_ranked_hypothesis_collection_with_matched_score(cur_hyp, list_hyp_info):
        for cur_hyp_info in list_hyp_info:
            if cur_hyp == cur_hyp_info[0]:
                cur_matched_score = int(cur_hyp_info[6][0])
                return cur_matched_score
        return -1

    bkg_q_list, dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2idx, dict_idx2bkg, dict_bkg2reasoningprocess = load_chem_annotation(chem_annotation_path, if_use_strict_survey_question=1)  
    # rank_collection_cnt_matched_insp: {cnt_matched_insp: [rank_ratio0, rank_ratio1, ...], ...}
    rank_collection_cnt_matched_insp = {}
    # cnt_insp_collection_cnt_matched_insp: {cnt_matched_insp: [cnt_total_used_insp0, cnt_total_used_insp1], ...}
    cnt_insp_collection_cnt_matched_insp = {}
    # rank_collection_matched_score: {matched_score(-1: no gdth insp): [rank_ratio0, rank_ratio1, ...], ...}
    rank_collection_matched_score = {}
    # matched_score_collection: {bkg_id: [matched_score0, matched_score1, ...], ...}
    matched_score_collection = {}
    for cur_id in range(data_id_range[0], data_id_range[1]+1):
        # cur_bkg_q_ori, cur_gdth_insps
        cur_bkg_q_ori = bkg_q_list[cur_id]
        cur_gdth_insps = dict_bkg2insp[cur_bkg_q_ori]
        # load file
        cur_file_path = file_root_name_path + str(cur_id) + ".json"
        with open(cur_file_path, 'r') as f:
            cur_data = json.load(f)
        # ranked_hypothesis_collection: {backgroud_question: ranked_hypothesis, ...}
        #   ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score, in descending order)
        ranked_hypothesis_collection = cur_data[0]
        if not if_random_order:
            # rank ranked_hypothesis_collection by average score again, in case it is not ranked (in descending order)
            for cur_bkg_q in ranked_hypothesis_collection.keys():
                ranked_hypothesis_collection[cur_bkg_q] = sorted(ranked_hypothesis_collection[cur_bkg_q], key=lambda x: x[1], reverse=True)
        else:
            # random shuffle
            for cur_bkg_q in ranked_hypothesis_collection.keys():
                random.shuffle(ranked_hypothesis_collection[cur_bkg_q])
        # ranked_hypothesis_collection_with_matched_score: {backgroud_question: ranked_hypothesis_matched_score, ...}
        #   ranked_hypothesis_matched_score: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id], matched_score, matched_score_reason], ...] (here core_insp_title is the matched groundtruth inspiration paper title) (sorted by average score, in descending order)
        ranked_hypothesis_collection_with_matched_score = cur_data[1]
        cur_bkg_q_key_list = list(ranked_hypothesis_collection.keys())
        assert len(cur_bkg_q_key_list) == 1
        for cur_bkg_q in ranked_hypothesis_collection.keys():
            len_gene_hyp_for_this_bkg_q = len(ranked_hypothesis_collection[cur_bkg_q])
            keep_top_len = min(int(len_gene_hyp_for_this_bkg_q * keep_top_ratio) + 1, len_gene_hyp_for_this_bkg_q)
            for cur_ranked_id in range(keep_top_len):
                cur_hyp = ranked_hypothesis_collection[cur_bkg_q][cur_ranked_id][0]
                cur_matched_score = check_whether_an_ranked_hypothesis_collection_item_in_ranked_hypothesis_collection_with_matched_score(cur_hyp, ranked_hypothesis_collection_with_matched_score[cur_bkg_q])
                cur_used_mutation_ids = ranked_hypothesis_collection[cur_bkg_q][cur_ranked_id][5]
                cur_round_id = ranked_hypothesis_collection[cur_bkg_q][cur_ranked_id][4]
                # if cur_round_id == 0 or cur_round_id == 3:
                #     print("cur_round_id: ", cur_round_id)
                if max_step != -1 and max_step < cur_round_id:
                    continue
                # cur_found_insps
                cur_found_insps = []
                for cur_mut_id in cur_used_mutation_ids:
                    if ";" in cur_mut_id:
                        cur_found_insps += cur_mut_id.split(";")
                    else:
                        cur_found_insps.append(cur_mut_id)
                cur_found_insps = get_rid_of_mutation_ids_in_found_insps(cur_found_insps)
                # cnt_intersection
                cnt_intersection = count_intersection_with_jaccard_similarity(cur_found_insps, cur_gdth_insps)
                cur_rank_ratio = (cur_ranked_id + 0.7) / len_gene_hyp_for_this_bkg_q
                # rank_collection_cnt_matched_insp
                if cnt_intersection not in rank_collection_cnt_matched_insp:
                    rank_collection_cnt_matched_insp[cnt_intersection] = []
                rank_collection_cnt_matched_insp[cnt_intersection].append(cur_rank_ratio)
                # cnt_insp_collection_cnt_matched_insp
                if cnt_intersection not in cnt_insp_collection_cnt_matched_insp:
                    cnt_insp_collection_cnt_matched_insp[cnt_intersection] = []
                cnt_insp_collection_cnt_matched_insp[cnt_intersection].append(len(cur_found_insps))
                # rank_collection_matched_score
                if cur_matched_score not in rank_collection_matched_score:
                    rank_collection_matched_score[cur_matched_score] = []
                rank_collection_matched_score[cur_matched_score].append(cur_rank_ratio)
                # matched_score_collection:
                if cur_id not in matched_score_collection:
                    matched_score_collection[cur_id] = []
                if cur_matched_score != -1:
                    matched_score_collection[cur_id].append(cur_matched_score)
                else:
                    # eval all hyp generated without any gdth insp as 0 point
                    # matched_score_collection[cur_id].append(0)
                    # skip all hyp generated without any gdth insp
                    pass
            # matched_score_collection[cur_id] shouldn't be empty; if empty, it means no hypotheses with positive Matched Score can be found for this bkg_id
            if len(matched_score_collection[cur_id]) == 0:
                matched_score_collection[cur_id].append(0)
    # sort matched_score_collection in descending order
    for cur_id in matched_score_collection.keys():
        matched_score_collection[cur_id] = sorted(matched_score_collection[cur_id], reverse=True)
                

    # len_rank_collection_cnt_matched_insp, ave_cnt_insp_collection_cnt_matched_insp, ave_rank_collection_cnt_matched_insp
    len_rank_collection_cnt_matched_insp = {tmp_key: len(tmp_value) for tmp_key, tmp_value in rank_collection_cnt_matched_insp.items()}
    print("len_rank_collection_cnt_matched_insp: ", len_rank_collection_cnt_matched_insp)
    ave_cnt_insp_collection_cnt_matched_insp = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in cnt_insp_collection_cnt_matched_insp.items()}
    print("ave_cnt_insp_collection_cnt_matched_insp: ", {k:round(v,3) for k,v in ave_cnt_insp_collection_cnt_matched_insp.items()})
    ave_rank_collection_cnt_matched_insp = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in rank_collection_cnt_matched_insp.items()}
    print("ave_rank_collection_cnt_matched_insp: ", {k:round(v,3) for k,v in ave_rank_collection_cnt_matched_insp.items()})

    # len_rank_collection_matched_score, ave_rank_collection_matched_score
    len_rank_collection_matched_score = {tmp_key: len(tmp_value) for tmp_key, tmp_value in rank_collection_matched_score.items()}
    print("\nlen_rank_collection_matched_score: ", len_rank_collection_matched_score)
    ave_rank_collection_matched_score = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in rank_collection_matched_score.items()}
    print("ave_rank_collection_matched_score: ", {k:round(v,3) for k,v in ave_rank_collection_matched_score.items()})

    # top_matched_score_collection: the top matched score for each background
    top_matched_score_collection = {tmp_key: tmp_value[0] for tmp_key, tmp_value in matched_score_collection.items()}
    # ave_matched_score_collection: the average matched score for each background
    ave_matched_score_collection = {tmp_key: np.mean(tmp_value) for tmp_key, tmp_value in matched_score_collection.items()}
    # cnt_top_matched_score_collection: {top_matched_score: cnt, ...}
    cnt_top_matched_score_collection = {}
    for cur_k, cur_v in top_matched_score_collection.items():
        if cur_v not in cnt_top_matched_score_collection:
            cnt_top_matched_score_collection[cur_v] = 0
        cnt_top_matched_score_collection[cur_v] += 1
    # cnt_ave_matched_score_collection: {ave_matched_score: cnt, ...}
    cnt_ave_matched_score_collection = {}
    for cur_k, cur_v in ave_matched_score_collection.items():
        cur_v_decimal = cur_v % 1
        cur_v_round = int(cur_v) + 1 if cur_v_decimal >= 0.5 else int(cur_v)
        if cur_v_round not in cnt_ave_matched_score_collection:
            cnt_ave_matched_score_collection[cur_v_round] = 0
        cnt_ave_matched_score_collection[cur_v_round] += 1
    # cnt_every_matched_score_collection: {matched_score: cnt, ...}
    cnt_every_matched_score_collection = {}
    for cur_id, cur_matched_scores in matched_score_collection.items():
        for cur_matched_score in cur_matched_scores:
            if cur_matched_score not in cnt_every_matched_score_collection:
                cnt_every_matched_score_collection[cur_matched_score] = 0
            cnt_every_matched_score_collection[cur_matched_score] += 1

    print("\ncnt_top_matched_score_collection: ", {k:round(v,3) for k,v in cnt_top_matched_score_collection.items()})
    print("cnt_ave_matched_score_collection: ", {k:round(v,3) for k,v in cnt_ave_matched_score_collection.items()})
    print("cnt_every_matched_score_collection: ", {k:round(v,3) for k,v in cnt_every_matched_score_collection.items()})
    ave_top_matched_score = np.mean([tmp_value for tmp_value in top_matched_score_collection.values()])
    ave_ave_matched_score = np.mean([tmp_value for tmp_value in ave_matched_score_collection.values()])
    print("\nave_top_matched_score: {:.3f}; ave_ave_matched_score: {:.3f}".format(ave_top_matched_score, ave_ave_matched_score))
    

## Function
#   Calculate the agreement (1) between the model and the expert, and (2) between experts
#   (1): only set expert_eval_file_path to the expert file path, and set second_expert_eval_file_path to None
#   (2): set both expert_eval_file_path and second_expert_eval_file_path to the expert file path to compare
# expert_eval_file: {bkg_id: {q_id: [gene_hyp, gdth_hyp, cnt_matched_insp, cur_matched_score, cur_matched_score_reason, expert_matched_score]}}
# second_expert_eval_file_path: if not None, then compare the matched score between two experts, else only compare the matched score between the model and the expert
def read_expert_eval_results(expert_eval_file_path, second_expert_eval_file_path=None):
    with open(expert_eval_file_path, "r") as f:
        expert_eval_file = json.load(f)
    if second_expert_eval_file_path != None:
        with open(second_expert_eval_file_path, "r") as f:
            second_expert_eval_file = json.load(f)
    seperate_bkg_id = 30
    num_q_per_bkg = 4

    if "Wanhao" in expert_eval_file_path:
        id_bkg_list = [str(i) for i in range(0, seperate_bkg_id)]
        assert expert_eval_file['19'][-1][5] == 3
    elif "Ben" in expert_eval_file_path:
        id_bkg_list = [str(i) for i in range(seperate_bkg_id, 51)]
    elif "Penghui" in expert_eval_file_path:
        # agreement between the model and the expert
        if second_expert_eval_file_path == None:
            id_bkg_list = [str(i) for i in range(0, 6)] + [str(i) for i in range(seperate_bkg_id, seperate_bkg_id+6)]
        else:
            # agreement between the experts
            if "Wanhao" in second_expert_eval_file_path:
                id_bkg_list = [str(i) for i in range(0, 6)]
            elif "Ben" in second_expert_eval_file_path:
                id_bkg_list = [str(i) for i in range(seperate_bkg_id, seperate_bkg_id+6)]
            else:
                raise ValueError("Invalid second_expert_eval_file_path")
    else:
        raise ValueError("Invalid name")

    # top_matched_score: {matched_score_expert: cnt, ...}
    top_matched_score_expert_collection = {}
    hard_consistency_score, soft_consistency_score = 0, 0
    # the number of matched score counted
    num_ms_cnted = 0
    for cur_bkg_id in id_bkg_list:
        # print("cur_bkg_id: ", cur_bkg_id)
        assert len(expert_eval_file[cur_bkg_id]) == num_q_per_bkg or len(expert_eval_file[cur_bkg_id]) == 0, print("len(expert_eval_file[cur_bkg_id]): ", len(expert_eval_file[cur_bkg_id]))
        if len(expert_eval_file[cur_bkg_id]) == 0:
            continue
        top_matched_score_expert = 0
        for cur_q_id in range(len(expert_eval_file[cur_bkg_id])):
            
            # print("expert_eval_file[cur_bkg_id][cur_q_id]: ", expert_eval_file[cur_bkg_id][cur_q_id])
            assert len(expert_eval_file[cur_bkg_id][cur_q_id]) == 6, print("cur_bkg_id: {}; cur_q_id: {}".format(cur_bkg_id, cur_q_id))
            if second_expert_eval_file_path == None:
                # score from gpt4o
                cur_auto_score = int(expert_eval_file[cur_bkg_id][cur_q_id][3])
            else:
                # score from another expert
                cur_auto_score = int(second_expert_eval_file[cur_bkg_id][cur_q_id][5])
            cur_expt_score = int(expert_eval_file[cur_bkg_id][cur_q_id][5])
            # print("cur_auto_score: {}; cur_expt_score: {}".format(cur_auto_score, cur_expt_score))
            if cur_auto_score == cur_expt_score:
                hard_consistency_score += 1
                soft_consistency_score += 1
            elif np.abs(cur_auto_score - cur_expt_score) <= 1:
                soft_consistency_score += 1
            # top_matched_score_expert
            if cur_expt_score > top_matched_score_expert:
                top_matched_score_expert = cur_expt_score
            num_ms_cnted += 1
        if top_matched_score_expert not in top_matched_score_expert_collection:
            top_matched_score_expert_collection[top_matched_score_expert] = 0
        top_matched_score_expert_collection[top_matched_score_expert] += 1

    print("num_ms_cnted: ", num_ms_cnted)
    assert num_ms_cnted == (len(id_bkg_list) * num_q_per_bkg)

    hard_consistency_score /= (len(id_bkg_list) * num_q_per_bkg)
    soft_consistency_score /= (len(id_bkg_list) * num_q_per_bkg)
    print("hard_consistency_score: {:.3f}; soft_consistency_score: {:.3f}".format(hard_consistency_score, soft_consistency_score))
    print("top_matched_score_expert_collection: {}".format(top_matched_score_expert_collection))
    
    
## FUNCTION:
#   give a evaluation file, and the selected hyp idex, find the full reasoning trace of that selected hyp
# Output
#   all_steps_idx: [selected_hyp_idx, (optional) prev index of selected_hyp_idx, (optional) prev prev index of selected_hyp_idx]
def find_full_reasoning_line(eval_file_dir, bkg_idx=0, selected_hyp_idx=0):
    eval_file_dir = eval_file_dir + str(bkg_idx) + ".json"
    with open(eval_file_dir, 'r') as f:
        d = json.load(f)
    b = list(d[0].keys())[0]
    # select a hyp to find its source; here we just use the first hyp
    selected_hyp = d[0][b][selected_hyp_idx][0]
    insp_trace = d[0][b][selected_hyp_idx][5]

    all_hyp_insp_trace = [sorted(d[0][b][cur_id][5]) for cur_id in range(len(d[0][b]))]

    # OUTPUT
    #   None (if cur_insp_trace represents the first step)
    #   [potential_prev_insp_trace_0, ...] (if cur_insp_trace represents the second or the third step)
    def obtain_prev_step_hyp_insp_trace(cur_insp_trace):
        prev_insp_trace_list = []
        # check if the third step
        if 'inter_recom_2' in cur_insp_trace:
            prev_insp_trace = copy.deepcopy(cur_insp_trace)
            prev_insp_trace.remove('inter_recom_2')
            potential_this_step_insp_list = []
            clustered_insp = None
            for cur_d in prev_insp_trace:
                if ';' not in cur_d:
                    potential_this_step_insp_list.append(cur_d)
                else:
                    assert clustered_insp == None
                    clustered_insp = cur_d
            assert clustered_insp != None, print("cur_insp_trace: ", cur_insp_trace)
            clustered_insp_split = clustered_insp.split(';')
            prev_insp_trace.remove(clustered_insp)
            prev_insp_trace += clustered_insp_split
            for cur_insp in potential_this_step_insp_list:
                cur_prev_insp_trace = copy.deepcopy(prev_insp_trace)
                # print("cur_prev_insp_trace: ", cur_prev_insp_trace)
                # print("cur_insp: ", cur_insp)
                cur_prev_insp_trace.remove(cur_insp)
                prev_insp_trace_list.append(cur_prev_insp_trace)
            return prev_insp_trace_list
        # check if the second step
        elif 'inter_recom_1' in cur_insp_trace:
            prev_insp_trace = copy.deepcopy(cur_insp_trace)
            prev_insp_trace.remove('inter_recom_1')
            potential_this_step_insp_list = []
            for cur_d in prev_insp_trace:
                # 0; 1; 2; recom
                if len(cur_d) > 6:
                    potential_this_step_insp_list.append(cur_d)
            for cur_insp in potential_this_step_insp_list:
                cur_prev_insp_trace = copy.deepcopy(prev_insp_trace)
                cur_prev_insp_trace.remove(cur_insp)
                prev_insp_trace_list.append(cur_prev_insp_trace)
            return prev_insp_trace_list
        # the first step
        else:
            return None
            
    cur_insp_trace = insp_trace
    # all_steps_idx: [this step idx, prev step idx, prev prev step idx]
    all_steps_idx = [selected_hyp_idx]
    while True:
        prev_insp_list = obtain_prev_step_hyp_insp_trace(cur_insp_trace)
        if prev_insp_list == None:
            break
        # find prev_step_index
        prev_step_index = None
        for cur_prev_insp in prev_insp_list:
            cur_prev_insp = sorted(cur_prev_insp)
            if cur_prev_insp in all_hyp_insp_trace:
                prev_step_index = all_hyp_insp_trace.index(cur_prev_insp)
                all_steps_idx.append(prev_step_index)
                break
        if prev_step_index == None:
            print("prev_insp_list: \n", prev_insp_list)
            print("\ncur_insp_trace: \n", cur_insp_trace)
        assert prev_step_index != None
        cur_insp_trace = d[0][b][prev_step_index][5]
    return all_steps_idx




# FUNCTION:
#   find the effect of EU by understanding the high matched scored hypothesis from non-EU branch, all EU branch, and only recom branch
def analyze_EU_find_proportion(eval_file_dir, start_bkg_idx=0, end_bkg_idx=51, threshold=4):
    total_non_eu_scores = []
    total_eu_scores = []
    total_recom_scores = []
    for cur_bkg_idx in range(start_bkg_idx, end_bkg_idx):
        with open(eval_file_dir+str(cur_bkg_idx)+".json", 'r') as f:
            full_data = json.load(f)
            b = list(full_data[1].keys())[0]
        cur_non_eu_scores = []
        cur_eu_scores = []
        cur_recom_scores = []
        for cur_id in range(len(full_data[1][b])):
            cur_insps = full_data[1][b][cur_id][5]
            cur_matched_score = int(full_data[1][b][cur_id][6][0])
            if cur_matched_score < threshold:
                continue
            if 'inter_recom_1' in cur_insps or 'inter_recom_2' in cur_insps:
                continue
            # print("cur_insps: ", cur_insps)
            # print("cur_matched_score: ", cur_matched_score)
            if '0' in cur_insps:
                # non EU
                cur_non_eu_scores.append(cur_matched_score)
            else:
                # EU
                cur_eu_scores.append(cur_matched_score)
            # only recom branch
            if 'recom' in cur_insps:
                cur_recom_scores.append(cur_matched_score)
        total_non_eu_scores += cur_non_eu_scores
        total_eu_scores += cur_eu_scores
        total_recom_scores += cur_recom_scores

    # total_non_eu_scores = sorted(total_non_eu_scores, reverse=True)
    # total_eu_scores = sorted(total_eu_scores, reverse=True)
    # total_recom_scores = sorted(total_recom_scores, reverse=True)
    mean_non_eu_scores = np.mean(total_non_eu_scores)
    mean_eu_scores = np.mean(total_eu_scores)
    mean_recom_scores = np.mean(total_recom_scores)

    print("len(total_non_eu_scores): {}; len(total_eu_scores): {}; len(total_recom_scores): {}".format(len(total_non_eu_scores), len(total_eu_scores), len(total_recom_scores)))
    print("mean_non_eu_scores: {:.3f}; mean_eu_scores: {:.3f}; mean_recom_scores: {:.3f}".format(mean_non_eu_scores, mean_eu_scores, mean_recom_scores))





if __name__ == "__main__":
    
    # check_moosechem_output()

    # check_difference_inspiration_retrieval_similarity_retrieval()

    ## Assumption 1
    # coarse_inspiration_search_gpt4_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_
    # coarse_inspiration_search_llama318b_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_
    # file_root_name_path = "./Checkpoints/coarse_inspiration_search_llama3170b_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_"
    # data_id_range = [0, 50]
    # # round_id = 3
    # for round_id in range(0, 4):
    #     get_average_screened_insp_hit_ratio_from_a_series_of_files(file_root_name_path, data_id_range, round_id)


    ## Assumption 2
    # (gdth insp; MOOSE-Chem) evaluation_gpt4_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_
    # (gdth insp; MOOSE-Chem, claude-3.5-Sonnet eval) evaluation_claude35S_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_
    # (gdth insp; MOOSE-Chem, without significance feedback (baseline 3); claude-3.5-Sonnet eval) evaluation_claude35S_baseline_3_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_
    # (full insp; MOOSE-chem) evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_1_interEA_1_beamsize_15_bkgid_
    # (full insp; MOOSE-Chem; claude-3.5-Sonnet eval) evaluation_claude35S_baseline_0_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_1_interEA_1_beamsize_15_bkgid_
    # file_root_name_path = "./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_"
    # data_id_range = [0, 50]
    # get_expert_eval_file_type = 0
    # if_save = False
    # if_not_only_from_gdth_insp = True
    # get_top_matched_score_for_each_background(file_root_name_path, data_id_range, get_expert_eval_file_type=get_expert_eval_file_type, if_save=if_save, if_not_only_from_gdth_insp=if_not_only_from_gdth_insp)


    ## Assumption 3; Exp 7/8/9/10/11
    # evaluation_gpt4_intraEA_1_interEA_1_gdthInsp_1_bkgid_
    # (full insp; MOOSE-chem) evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_1_interEA_1_beamsize_15_bkgid_
    # (baseline) evaluation_gpt4_baseline_2_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_0_interEA_0_beamsize_15_bkgid_
    file_root_name_path = "./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_1_interEA_1_beamsize_15_bkgid_"
    data_id_range = [0, 50]
    if_random_order = False
    keep_top_ratio = 1.0
    max_step = 1
    get_average_ranking_position_for_hyp_with_gdth_insp(file_root_name_path, data_id_range, if_random_order=if_random_order, keep_top_ratio=keep_top_ratio, max_step=max_step)


    ## expert eval
    # expert_eval_for_selected_hyp_in_exp_5_Wanhao
    # expert_eval_for_selected_hyp_in_exp_5_BenGao
    # expert_eval_for_selected_hyp_in_exp_5_Penghui
    # expert_eval_for_selected_hyp_in_exp_8_Wanhao
    # expert_eval_for_selected_hyp_in_exp_8_Ben
    # expert_eval_file_path = "./Expert_Evaluation/expert_eval_for_selected_hyp_in_exp_5_Wanhao.json"
    # # second_expert_eval_file_path = "./Expert_Evaluation/expert_eval_for_selected_hyp_in_exp_5_BenGao.json"
    # second_expert_eval_file_path = None
    # read_expert_eval_results(expert_eval_file_path, second_expert_eval_file_path=second_expert_eval_file_path)


    ## Analyze the full reasoning intermediate steps of a final step hypothesis
    # file_root_name_path = "./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_0_interEA_1_beamsize_15_bkgid_"
    # all_steps_idx = find_full_reasoning_line(file_root_name_path, bkg_idx=0, selected_hyp_idx=0)
    # print("all_steps_idx: ", all_steps_idx)


    ## Find what proportion of high-scored (match score) hypothesis can be resulted from EU
    # file_root_name_path = "./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_roundInsp_1_intraEA_1_interEA_1_beamsize_15_bkgid_"
    # analyze_EU_find_proportion(file_root_name_path, start_bkg_idx=0, end_bkg_idx=51, threshold=0)