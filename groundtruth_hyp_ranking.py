import os, argparse, json, time, copy, math
from openai import OpenAI, AzureOpenAI
from utils import load_chem_annotation, instruction_prompts, llm_generation, pick_score
import numpy as np


class GroundTruth_Hyp_Ranking(object):
    def __init__(self, args) -> None:
        self.args = args
        # set OpenAI API key
        if args.api_type == 0:
            self.client = OpenAI(api_key=args.api_key, base_url="https://api.claudeshop.top/v1")
        elif args.api_type == 1:
            self.client = AzureOpenAI(
                azure_endpoint = "https://gd-sweden-gpt4vision.openai.azure.com/", 
                api_key=args.api_key,
                api_version="2024-06-01"
            )
        elif args.api_type == 2:
            self.client = AzureOpenAI(
                azure_endpoint = "https://declaregpt4.openai.azure.com/", 
                api_key=args.api_key,  
                api_version="2024-06-01"
            )
        else:
            raise NotImplementedError
        # groundtruth hypothesis
        self.bkg_q_list, self.dict_bkg2insp, self.dict_bkg2survey, self.dict_bkg2groundtruthHyp, self.dict_bkg2note, self.dict_bkg2idx, self.dict_idx2bkg, self.dict_bkg2reasoningprocess = load_chem_annotation(args.chem_annotation_path, self.args.if_use_strict_survey_question, self.args.if_use_background_survey)      
        

    ## INPUT
    # cur_hyp: text
    ## Output
    # score_collection: ['score0', 'score1', 'score2', 'score3']
    # score_reason_collection: ['reason0', 'reason1', 'reason2', 'reason3']
    def four_aspects_self_numerical_evaluation_for_hyp(self, cur_hyp):
        prompts = instruction_prompts("four_aspects_self_numerical_evaluation")
        assert len(prompts) == 2
        # cur_hypothesis_prompt: for evaluation, we only need the hypothesis itself, but not reasoning process
        cur_hypothesis_prompt = "hypothesis: {}.".format(cur_hyp)
        full_prompt = prompts[0] + cur_hypothesis_prompt + prompts[1]
        # generation
        while True:
            try:
                score_text = llm_generation(full_prompt, self.args.model_name, self.client, api_type=self.args.api_type)
                score_collection, score_reason_collection, if_successful = pick_score(score_text, full_prompt)
                assert if_successful == True
                break
            except AssertionError as e:
                # if the format
                print("AssertionError: {}, try again..".format(e))
            except Exception as e:
                print("Exception: {}, try again..".format(e))
        return score_collection, score_reason_collection


    ## input
    # cur_bkg: text
    # cur_score_collection: ['score0', 'score1', 'score2', 'score3']; scores for groundtruth hypothesis
    ## OUTPUT
    # final_ratio_overall_and_four_aspects: [[first_index_ratio, last_index_ratio], [first_index_ratio_validness, last_index_ratio_validness], [first_index_ratio_novelty, last_index_ratio_novelty], [first_index_ratio_significance, last_index_ratio_significance], [first_index_ratio_potential, last_index_ratio_potential]]
    #   first_index_ratio, last_index_ratio: [0, 1]
    def get_rank_ratio_for_each_hyp(self, cur_id_bkg, cur_bkg, cur_score_collection):
        # generated hypothesis and their R(h)
        with open(args.evaluate_result_dir + str(cur_id_bkg) + ".json", 'r') as f:
            full_evaluate_result = json.load(f)
        # ranked_hypothesis_collection: {backgroud_question: ranked_hypothesis, ...}
            #   ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score, in descending order)
        ranked_hypothesis_collection = full_evaluate_result[0]
        # cur_ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score, in descending order)
        cur_ranked_hypothesis = ranked_hypothesis_collection[cur_bkg]
        ave_score_groundtruth_hyp = np.mean(cur_score_collection)
        # print("cur_id_bkg: {}; ave_score_groundtruth_hyp: {}".format(cur_id_bkg, ave_score_groundtruth_hyp))
        # generated_hyp_ave_score_ranked_list should be sorted
        generated_hyp_ave_score_ranked_list = [cur_ranked_hypothesis[id][1] for id in range(len(cur_ranked_hypothesis))]
        # generated_hyp_validness_score_ranked_list
        generated_hyp_validness_score_ranked_list = [cur_ranked_hypothesis[id][2][0] for id in range(len(cur_ranked_hypothesis))]
        generated_hyp_validness_score_ranked_list = sorted(generated_hyp_validness_score_ranked_list, reverse=True)
        # generated_hyp_novelty_score_ranked_list
        generated_hyp_novelty_score_ranked_list = [cur_ranked_hypothesis[id][2][1] for id in range(len(cur_ranked_hypothesis))]
        generated_hyp_novelty_score_ranked_list = sorted(generated_hyp_novelty_score_ranked_list, reverse=True)
        # generated_hyp_significance_score_ranked_list
        generated_hyp_significance_score_ranked_list = [cur_ranked_hypothesis[id][2][2] for id in range(len(cur_ranked_hypothesis))]
        generated_hyp_significance_score_ranked_list = sorted(generated_hyp_significance_score_ranked_list, reverse=True)
        # generated_hyp_potential_score_ranked_list
        generated_hyp_potential_score_ranked_list = [cur_ranked_hypothesis[id][2][3] for id in range(len(cur_ranked_hypothesis))]
        generated_hyp_potential_score_ranked_list = sorted(generated_hyp_potential_score_ranked_list, reverse=True)
        # print("generated_hyp_ave_score_ranked_list: ", generated_hyp_ave_score_ranked_list)
        

        def get_first_last_ranking_index(ranked_list, value_to_rank):
            for idx_rank in range(len(ranked_list)):
                if_found_first_ranking_index = False
                first_index, last_index = None, None
                # prev_num
                if idx_rank == 0:
                    prev_num = ranked_list[idx_rank]
                cur_num = ranked_list[idx_rank]
                assert cur_num <= prev_num
                if if_found_first_ranking_index == False:
                    if value_to_rank == cur_num:
                        first_index = idx_rank
                        if_found_first_ranking_index = True
                    elif value_to_rank > cur_num:
                        first_index = idx_rank
                        if_found_first_ranking_index = True
                        last_index = idx_rank
                        return first_index, last_index
                else:
                    assert first_index != None
                    if value_to_rank > cur_num:
                        last_index = idx_rank
                        return first_index, last_index
            assert idx_rank == len(ranked_list) - 1
            if first_index == None:
                first_index = idx_rank
            last_index = idx_rank
            return first_index, last_index

        len_gene_and_groundtruth_hyp = len(cur_ranked_hypothesis) + 1
        first_index, last_index = get_first_last_ranking_index(generated_hyp_ave_score_ranked_list, ave_score_groundtruth_hyp)
        first_index_validness, last_index_validness = get_first_last_ranking_index(generated_hyp_validness_score_ranked_list, cur_score_collection[0])
        first_index_novelty, last_index_novelty = get_first_last_ranking_index(generated_hyp_novelty_score_ranked_list, cur_score_collection[1])
        first_index_significance, last_index_significance = get_first_last_ranking_index(generated_hyp_significance_score_ranked_list, cur_score_collection[2])
        first_index_potential, last_index_potential = get_first_last_ranking_index(generated_hyp_potential_score_ranked_list, cur_score_collection[3])
        
        # first_index_ratio, first_index_ratio for average score from the four scores
        first_index_ratio = first_index / len_gene_and_groundtruth_hyp
        last_index_ratio = last_index / len_gene_and_groundtruth_hyp
        # first_index_ratio, first_index_ratio for each score
        first_index_ratio_validness = first_index_validness / len_gene_and_groundtruth_hyp
        last_index_ratio_validness = last_index_validness / len_gene_and_groundtruth_hyp
        first_index_ratio_novelty = first_index_novelty / len_gene_and_groundtruth_hyp
        last_index_ratio_novelty = last_index_novelty / len_gene_and_groundtruth_hyp
        first_index_ratio_significance = first_index_significance / len_gene_and_groundtruth_hyp
        last_index_ratio_significance = last_index_significance / len_gene_and_groundtruth_hyp
        first_index_ratio_potential = first_index_potential / len_gene_and_groundtruth_hyp
        last_index_ratio_potential = last_index_potential / len_gene_and_groundtruth_hyp

        final_ratio_overall_and_four_aspects = [[first_index_ratio, last_index_ratio], [first_index_ratio_validness, last_index_ratio_validness], [first_index_ratio_novelty, last_index_ratio_novelty], [first_index_ratio_significance, last_index_ratio_significance], [first_index_ratio_potential, last_index_ratio_potential]]
        for cur_ratio_idx in range(len(final_ratio_overall_and_four_aspects)):
            cur_ave_ratio = np.mean(final_ratio_overall_and_four_aspects[cur_ratio_idx])
            final_ratio_overall_and_four_aspects[cur_ratio_idx].append(cur_ave_ratio)
        print("cur_id_bkg: {}; cur_ave_ratio_overall: {:.2f}; cur_ave_ratio_validness: {:.2f}; cur_ave_ratio_novelty: {:.2f}; cur_ave_ratio_significance: {:.2f}; cur_ave_ratio_potential: {:.2f}".format(cur_id_bkg, final_ratio_overall_and_four_aspects[0][2], final_ratio_overall_and_four_aspects[1][2], final_ratio_overall_and_four_aspects[2][2], final_ratio_overall_and_four_aspects[3][2], final_ratio_overall_and_four_aspects[4][2]))
        return final_ratio_overall_and_four_aspects



    def looping(self):
        # groundtruthHyp_fourScores_collection: [[cur_score_collection, cur_score_reason_collection, final_ratio_overall_and_four_aspects], ...]
        #   final_ratio_overall_and_four_aspects: [[first_ratio, last_ratio, ave_ratio], ...] (average score, validness score, novelty score, significance score, potential score)
        groundtruthHyp_fourScores_collection = []
        ave_index_ratio_list = []
        for cur_id_bkg in range(len(self.bkg_q_list)):
            cur_bkg = self.bkg_q_list[cur_id_bkg]
            cur_hyp = self.dict_bkg2groundtruthHyp[cur_bkg]
            # get scores for cur_hyp
            cur_score_collection, cur_score_reason_collection = self.four_aspects_self_numerical_evaluation_for_hyp(cur_hyp)
            # print("cur_score_collection: ", cur_score_collection)
            final_ratio_overall_and_four_aspects = self.get_rank_ratio_for_each_hyp(cur_id_bkg, cur_bkg, cur_score_collection)
            groundtruthHyp_fourScores_collection.append([cur_id_bkg, cur_score_collection, cur_score_reason_collection, final_ratio_overall_and_four_aspects])
            ave_index_ratio_list.append(final_ratio_overall_and_four_aspects[0][2])
        ave_ave_index_ratio = np.mean(ave_index_ratio_list)
        # save
        if args.if_save:
            with open(args.output_dir, 'w') as f:
                json.dump(groundtruthHyp_fourScores_collection, f)
        return ave_ave_index_ratio

        
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hypothesis generation')
    parser.add_argument("--model_name", type=str, default="claude35S", help="model name: gpt4/chatgpt/chatgpt16k/claude35S/gemini15P")
    parser.add_argument("--api_type", type=int, default=0, help="2: use Soujanya's API; 1: use Dr. Xie's API; 0: use api from shanghai ai lab")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--chem_annotation_path", type=str, default="./Data/chem_research_2024.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--if_use_background_survey", type=int, default=1, help="whether use background survey. 0: not use (replace the survey as 'Survey not provided. Please overlook the survey.'); 1: use")
    parser.add_argument("--if_use_strict_survey_question", type=int, default=1, help="whether to use the strict version of background survey and background question. strict version means the background should not have any close information to inspirations and the hypothesis, even if the close information is a commonly used method in that particular background question domain.")
    parser.add_argument("--evaluate_result_dir", type=str, default="./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_")
    parser.add_argument("--if_save", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./Checkpoints/groundtruth_hypothesis_automatic_scores_four_aspects.json")
    args = parser.parse_args()

    assert args.if_save in [0, 1]
    if not os.path.exists(args.output_dir):
        gtr = GroundTruth_Hyp_Ranking(args)
        ave_ave_index_ratio = gtr.looping()
    else:
        # groundtruthHyp_fourScores_collection: [[cur_id_bkg, cur_score_collection, cur_score_reason_collection, final_ratio_overall_and_four_aspects], ...]
        #   final_ratio_overall_and_four_aspects: [[first_ratio, last_ratio, ave_ratio], ...] (average score, validness score, novelty score, significance score, potential score)
        print("{} already exists.".format(args.output_dir))
        with open(args.output_dir, 'r') as f:
            groundtruthHyp_fourScores_collection = json.load(f)
        ave_index_ratio_list = [groundtruthHyp_fourScores_collection[id][3][0][2] for id in range(len(groundtruthHyp_fourScores_collection))]
        ave_index_ratio_validness_list = [groundtruthHyp_fourScores_collection[id][3][1][2] for id in range(len(groundtruthHyp_fourScores_collection))]
        ave_index_ratio_novelty_list = [groundtruthHyp_fourScores_collection[id][3][2][2] for id in range(len(groundtruthHyp_fourScores_collection))]
        ave_index_ratio_significance_list = [groundtruthHyp_fourScores_collection[id][3][3][2] for id in range(len(groundtruthHyp_fourScores_collection))]
        ave_index_ratio_potential_list = [groundtruthHyp_fourScores_collection[id][3][4][2] for id in range(len(groundtruthHyp_fourScores_collection))]

        ave_ave_index_ratio = np.mean(ave_index_ratio_list)
        ave_ave_index_ratio_validness = np.mean(ave_index_ratio_validness_list)
        ave_ave_index_ratio_novelty = np.mean(ave_index_ratio_novelty_list)
        ave_ave_index_ratio_significance = np.mean(ave_index_ratio_significance_list)
        ave_ave_index_ratio_potential = np.mean(ave_index_ratio_potential_list)

    print("ave_ave_index_ratio_overall: {:.2f}; ave_ave_index_ratio_validness: {:.2f}; ave_ave_index_ratio_novelty: {:.2f}; ave_ave_index_ratio_significance: {:.2f}; ave_ave_index_ratio_potential: {:.2f}".format(ave_ave_index_ratio, ave_ave_index_ratio_validness, ave_ave_index_ratio_novelty, ave_ave_index_ratio_significance, ave_ave_index_ratio_potential))

