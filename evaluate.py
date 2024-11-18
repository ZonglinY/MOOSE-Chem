import os, argparse, json, time, copy, math
import numpy as np
from openai import OpenAI, AzureOpenAI
from utils import load_chem_annotation, instruction_prompts, llm_generation_while_loop, recover_generated_title_to_exact_version_of_title, load_dict_title_2_abstract, if_element_in_list_with_similarity_threshold

class Evaluate(object):

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
        # annotated bkg research question and its annotated groundtruth inspiration paper titles
        self.bkg_q_list, self.dict_bkg2insp, self.dict_bkg2survey, self.dict_bkg2groundtruthHyp, self.dict_bkg2note, self.dict_bkg2idx, self.dict_idx2bkg, self.dict_bkg2reasoningprocess = load_chem_annotation(args.chem_annotation_path, self.args.if_use_strict_survey_question)   
        # dict_title_2_abstract: {'title': 'abstract', ...}
        self.dict_title_2_abstract = load_dict_title_2_abstract(title_abstract_collector_path=args.title_abstract_all_insp_literature_path)  
        ## load raw hypothesis
        # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
        #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom'
        with open(args.hypothesis_dir, 'r') as f:
            self.final_data_collection = json.load(f)
        

    def run(self):
        ## obtain ranked_hypothesis_collection and ranked_hypothesis_collection_with_matched_score
        if self.args.if_load_from_saved:
            with open(self.args.output_dir, 'r') as f:
                self.ranked_hypothesis_collection, self.ranked_hypothesis_collection_with_matched_score, self.matched_insp_hyp_collection = json.load(f)
                print("Loaded data from ", self.args.output_dir)
        else:
            ## hypothesis ranking
            # ranked_hypothesis_collection: {backgroud_question: ranked_hypothesis, ...}
            #   ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score, in descending order)
            self.ranked_hypothesis_collection = self.hypothesis_ranking(self.final_data_collection)
            # ranked_hypothesis_collection_with_matched_score: {backgroud_question: ranked_hypothesis_matched_score, ...}
            #   ranked_hypothesis_matched_score: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id], [matched_score, matched_score_reason]], ...] (here core_insp_title is the matched groundtruth inspiration paper title) (sorted by average score, in descending order)
            self.ranked_hypothesis_collection_with_matched_score = self.automatic_evaluation_by_reference(self.ranked_hypothesis_collection)

        ## analysis
        # print rank based on the number of matched inspirations
        # matched_insp_hyp_collection: [[cur_hyp, cur_gdth_hyp, cur_ave_score, cur_scores, cnt_matched_insp, cur_used_insps_set, cur_full_gdth_insps, cur_matched_score, cur_matched_score_reason, cur_round_id], ...] (sorted by cnt_matched_insp, in descending order)
        self.matched_insp_hyp_collection = self.analyse_gene_hyp_closest_to_gdth_hyp(self.ranked_hypothesis_collection_with_matched_score)

        ## save results
        if self.args.if_save == 1:
            with open(self.args.output_dir, 'w') as f:
                json.dump([self.ranked_hypothesis_collection, self.ranked_hypothesis_collection_with_matched_score, self.matched_insp_hyp_collection], f)
                print("Results saved to ", self.args.output_dir)


    ## Input
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom'
    #     hypthesis_mutation_collection['inter_com']: {core_insp_title_best_mutation_id: {matched_insp_title0: [[hyp0, reasoning process0, feedback0], ...], ...}}
    ## Output
    # ranked_hypothesis_collection: {backgroud_question: ranked_hypothesis, ...}
        #   ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score)
    ## Q: do not consider 'self_explore' now; can attend to unlimited steps of inter-EA recombination
    def hypothesis_ranking(self, final_data_collection):
        # find the index of the first element in the list that is greater than the item (list is sorted in descending order)
        def find_index(lst, item):
            for i in range(len(lst)):
                if item > lst[i]:
                    return i
            return len(lst)
        # complete ranked_hypothesis_collection; no need to consider "self_explore"
        ranked_hypothesis_collection = {}
        for cur_background_question in final_data_collection.keys():
            ranked_hypothesis_collection[cur_background_question] = []
            for cur_core_insp_title in final_data_collection[cur_background_question].keys():
                for cur_mutation_id in final_data_collection[cur_background_question][cur_core_insp_title].keys():
                    if "inter_recom" not in cur_mutation_id and "self_explore" not in cur_mutation_id:
                        cur_hypothesis_collection = final_data_collection[cur_background_question][cur_core_insp_title][cur_mutation_id]
                        cur_hyp = cur_hypothesis_collection[-1][0]
                        cur_scores = cur_hypothesis_collection[-1][-1][0]
                        assert len(cur_scores) == 4
                        cur_ave_score = np.mean(cur_scores)
                        cur_round_id = 1
                        cur_index = find_index([x[1] for x in ranked_hypothesis_collection[cur_background_question]], cur_ave_score)
                        ranked_hypothesis_collection[cur_background_question].insert(cur_index, [cur_hyp, cur_ave_score, cur_scores, cur_core_insp_title, cur_round_id, [cur_core_insp_title, cur_mutation_id]])
                    elif "inter_recom" in cur_mutation_id:
                        # cur_hypothesis_collection: {core_insp_title_best_mutation_id: {matched_insp_title0: [[hyp0, reasoning process0, feedback0], ...], ...}}
                        cur_hypothesis_collection = final_data_collection[cur_background_question][cur_core_insp_title][cur_mutation_id]
                        for cur_core_insp_title_best_mutation_id in cur_hypothesis_collection.keys():
                            for cur_matched_insp_title in cur_hypothesis_collection[cur_core_insp_title_best_mutation_id].keys():
                                cur_data = cur_hypothesis_collection[cur_core_insp_title_best_mutation_id][cur_matched_insp_title]
                                cur_hyp = cur_data[-1][0]
                                cur_scores = cur_data[-1][-1][0]
                                assert len(cur_scores) == 4
                                cur_ave_score = np.mean(cur_scores)
                                cur_round_id = int(cur_mutation_id.strip().strip("inter_recom_")) + 1
                                cur_index = find_index([x[1] for x in ranked_hypothesis_collection[cur_background_question]], cur_ave_score)
                                ranked_hypothesis_collection[cur_background_question].insert(cur_index, [cur_hyp, cur_ave_score, cur_scores, cur_core_insp_title, cur_round_id, [cur_core_insp_title, cur_mutation_id, cur_core_insp_title_best_mutation_id, cur_matched_insp_title]])
        return ranked_hypothesis_collection
                                
        
    ## Function:
    # automatic evaluation by reference 
    #   only evaluate those hypotheses whose core_insp_title is in the groundtruth inspiration paper titles, and append matched score to ranked_hypothesis
    ## Input
    # ranked_hypothesis_collection: {backgroud_question: ranked_hypothesis, ...}
    #   ranked_hypothesis: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id]], ...] (sorted by average score, in descending order)
    ## Output
    # ranked_hypothesis_collection_with_matched_score: {backgroud_question: ranked_hypothesis_matched_score, ...}
    #   ranked_hypothesis_matched_score: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id], [matched_score, matched_score_reason]], ...] (here core_insp_title is the matched groundtruth inspiration paper title); ranked by ave_score
    def automatic_evaluation_by_reference(self, ranked_hypothesis_collection):
        ranked_hypothesis_collection_with_matched_score = {}
        for cur_background_question in ranked_hypothesis_collection.keys():
            ranked_hypothesis_collection_with_matched_score[cur_background_question] = []
            # print("Evaluating for background question: {}; total number of hypotheses: {}".format(cur_background_question, len(ranked_hypothesis_collection[cur_background_question])))
            for cur_id_hyp in range(len(ranked_hypothesis_collection[cur_background_question])):
                cur_hyp = ranked_hypothesis_collection[cur_background_question][cur_id_hyp][0]
                ## check whether cur_core_insp_title is in the groundtruth inspiration paper titles
                cur_core_insp_title = ranked_hypothesis_collection[cur_background_question][cur_id_hyp][3]
                # cur_groundtruth_insp_titles: [insp0, insp1, ...]
                cur_groundtruth_insp_titles = self.dict_bkg2insp[cur_background_question]
                # recover the groundtruth inspirations to the exact version of title (the ones in title_abstract.json, even chem_research_2024.xlsx is not counted as groundtruth here, since title_abstract.json might have conflicts with chem_research_2024.xlsx, and title_abstract.json is more complete, so we choose title_abstract.json as the groundtruth, although chem_research_2024.xlsx is our benchmark and title_abstract.json is only a processed intermediate file) 
                cur_groundtruth_insp_titles = [recover_generated_title_to_exact_version_of_title(list(self.dict_title_2_abstract.keys()), cur_gdth_insp) for cur_gdth_insp in cur_groundtruth_insp_titles]
                # to see whether cur_core_insp_title is in cur_groundtruth_insp_titles
                if_insp_in_groundtruth = if_element_in_list_with_similarity_threshold(cur_groundtruth_insp_titles, cur_core_insp_title, threshold=0.7)
                if if_insp_in_groundtruth == False:
                    continue
                ## start evaluation
                cur_groundtruth_hyp = self.dict_bkg2groundtruthHyp[cur_background_question]
                cur_keypoints = self.dict_bkg2note[cur_background_question]
                # cur_matched_score_and_reason: [matched_score, reason]
                cur_matched_score_and_reason = self.evaluate_for_one_hypothesis(cur_hyp, cur_groundtruth_hyp, cur_keypoints)
                ranked_hypothesis_collection_with_matched_score[cur_background_question].append(ranked_hypothesis_collection[cur_background_question][cur_id_hyp] + cur_matched_score_and_reason)
            print("Evaluating for background question: {}; total number of hypotheses: {}; number of hypotheses with matched score: {}".format(cur_background_question, len(ranked_hypothesis_collection[cur_background_question]), len(ranked_hypothesis_collection_with_matched_score[cur_background_question])))
        return ranked_hypothesis_collection_with_matched_score
                

    ## Function:
    # evaluate for one hypothesis by reference to get matched score
    ## Input
    # gene_hyp: str; gold_hyp: str
    ## Output
    # matched_score: int in 1-5 Likert scale
    def evaluate_for_one_hypothesis(self, gene_hyp, gold_hyp, keypoints):
        prompts = instruction_prompts('eval_matched_score')
        full_prompt = prompts[0] + gene_hyp + prompts[1] + gold_hyp + prompts[2] + keypoints + prompts[3]
        # structured_gene: [matched_score, reason]
        structured_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Matched score:', 'Reason:'], temperature=0.0, api_type=self.args.api_type)
        return structured_gene
        

    ## Function:
    # print the average score of those generated hypotheses with the most similar inspirations with the ground truth hypothesis
    ## Input:
    # ranked_hypothesis_collection_with_matched_score: {backgroud_question: ranked_hypothesis_matched_score, ...}
    #   ranked_hypothesis_matched_score: [[hyp, ave_score, scores, core_insp_title, round_id, [first_round_mutation_id, second_round_mutation_id], matched_score], ...] (here core_insp_title is the matched groundtruth inspiration paper title)
    ## Output:
    # matched_insp_hyp_collection: [[cur_hyp, cur_gdth_hyp, cur_ave_score, cur_scores, cnt_matched_insp, cur_used_insps_set, cur_full_gdth_insps, cur_matched_score, cur_matched_score_reason, cur_round_id], ...]
    def analyse_gene_hyp_closest_to_gdth_hyp(self, ranked_hypothesis_collection_with_matched_score):
        matched_insp_hyp_collection = []
        for cur_background_question in ranked_hypothesis_collection_with_matched_score.keys():
            for cur_id_hyp in range(len(ranked_hypothesis_collection_with_matched_score[cur_background_question])):
                cur_hyp = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][0]
                cur_ave_score = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][1]
                cur_scores = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][2]
                cur_round_id = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][4]
                cur_mutation_id_trail = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][5]
                cur_matched_score_reason = ranked_hypothesis_collection_with_matched_score[cur_background_question][cur_id_hyp][6]
                cur_gdth_hyp = self.dict_bkg2groundtruthHyp[cur_background_question]
                cur_used_insps = []
                for cur_mut in cur_mutation_id_trail:
                    if ";" in cur_mut:
                        cur_used_insps += cur_mut.split(";")
                    else:
                        cur_used_insps.append(cur_mut)
                cur_used_insps_set = list(set(cur_used_insps))
                # should be no repeated inspirations
                assert len(cur_used_insps_set) == len(cur_used_insps)
                cur_full_gdth_insps = self.dict_bkg2insp[cur_background_question]
                cnt_matched_insp = 0
                # print("cur_used_insps_set: ", cur_used_insps_set)
                for cur_gdth_insp in cur_full_gdth_insps:
                    if if_element_in_list_with_similarity_threshold(cur_used_insps_set, cur_gdth_insp, threshold=0.7):
                        cnt_matched_insp += 1
                if cnt_matched_insp > 0:
                    matched_insp_hyp_collection.append([cur_hyp, cur_gdth_hyp, cur_ave_score, cur_scores, cnt_matched_insp, cur_used_insps_set, cur_full_gdth_insps, cur_matched_score_reason[0], cur_matched_score_reason[1], cur_round_id])
        # rank matched_insp_hyp_collection based on cnt_matched_insp
        matched_insp_hyp_collection = sorted(matched_insp_hyp_collection, key=lambda x: x[4], reverse=True)
        for cur_matched_insp_hyp in matched_insp_hyp_collection:
            # print("cnt_matched_insp: {}; ave_score: {}; matched_score: {}; \n\ngene_hyp: \n{}; \n\ngdth_hyp: \n{}".format(cur_matched_insp_hyp[4], cur_matched_insp_hyp[2], cur_matched_insp_hyp[7], cur_matched_insp_hyp[0], cur_matched_insp_hyp[1]))
            print("cnt_matched_insp: {}; ave_score: {}; matched_score: {}".format(cur_matched_insp_hyp[4], cur_matched_insp_hyp[2], cur_matched_insp_hyp[7]))
        return matched_insp_hyp_collection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypothesis evaluation by reference')
    parser.add_argument("--model_name", type=str, default="chatgpt", help="model name: gpt4/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=1, help="1: use Dr. Xie's API; 0: use api from shanghai ai lab")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--chem_annotation_path", type=str, default="./chem_research_2024.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--if_use_strict_survey_question", type=int, default=1, help="whether to use the strict version of background survey and background question. strict version means the background should not have any close information to inspirations and the hypothesis, even if the close information is a commonly used method in that particular background question domain.")
    parser.add_argument("--title_abstract_all_insp_literature_path", type=str, default="./title_abstract.json")
    parser.add_argument("--hypothesis_dir", type=str, default="./Checkpoints/hypothesis_generation_gpt4_bkgid_0.json")
    parser.add_argument("--output_dir", type=str, default="./Checkpoints/hypothesis_evaluation_results.json")
    parser.add_argument("--if_save", type=int, default=0, help="whether save grouping results")
    parser.add_argument("--if_load_from_saved", type=int, default=0, help="whether load data that is previous to inter-EA recombination; when used, the framework will load data from output_dir, instead of generating from scratch; mainly used for debugging and improving inter-EA recombination") 
    parser.add_argument("--corpus_size", type=int, default=300, help="the number of total inspiration (paper) corpus (both groundtruth insp papers and non-groundtruth insp papers)")
    args = parser.parse_args()

    assert args.model_name in ['chatgpt', 'chatgpt16k', 'gpt4', 'claude35S', 'gemini15P', 'llama318b', 'llama3170b', 'llama31405b']
    assert args.api_type in [0, 1, 2]
    assert args.if_use_strict_survey_question in [0, 1]
    assert args.if_save in [1]
    assert args.if_load_from_saved in [0, 1]
    # change args.title_abstract_all_insp_literature_path
    assert args.title_abstract_all_insp_literature_path == "./title_abstract.json"
    args.title_abstract_all_insp_literature_path = './Data/Inspiration_Corpus_{}.json'.format(args.corpus_size)
    print("args: ", args)

    # skip if the output_dir already exists
    # Q: overlook args.if_load_from_saved for recent experiments
    if os.path.exists(args.output_dir):
        print("Warning: {} already exists.".format(args.output_dir))
    else:
        evaluate = Evaluate(args)
        evaluate.run()
    print("Evaluation finished.")
