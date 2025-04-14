from ast import Not
from multiprocessing import Value
import os, sys, argparse, json, time, copy, math, builtins
from openai import OpenAI, AzureOpenAI
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Method.utils import load_chem_annotation, load_dict_title_2_abstract, load_found_inspirations, get_item_from_dict_with_very_similar_but_not_exact_key, instruction_prompts, llm_generation, get_structured_generation_from_raw_generation, pick_score, llm_generation_while_loop, recover_generated_title_to_exact_version_of_title, load_groundtruth_inspirations_as_screened_inspirations, exchange_order_in_list
from Method.logging_utils import setup_logger

class HypothesisGenerationEA(object):
    # custom_rq (text) and custom_bs (text) are used when the user has their own research question and background survey to work on (but not those in the Tomato-Chem benchmark), and leverage MOOSE-Chem for inference
    def __init__(self, args, custom_rq=None, custom_bs=None) -> None:
        self.args = args
        self.custom_rq = custom_rq
        self.custom_bs = custom_bs
        ## Set API client
        # openai client
        if args.api_type == 0:
            self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        # azure client
        elif args.api_type == 1:
            self.client = AzureOpenAI(
                azure_endpoint = args.base_url, 
                api_key=args.api_key,  
                api_version="2024-06-01"
            )
        else:
            raise NotImplementedError
        ## Load research background: Use the research question and background survey in Tomato-Chem or the custom ones from input
        if custom_rq == None and custom_bs == None:
            # annotated bkg research question and its annotated groundtruth inspiration paper titles
            self.bkg_q_list, self.dict_bkg2insp, self.dict_bkg2survey, self.dict_bkg2groundtruthHyp, self.dict_bkg2note, self.dict_bkg2idx, self.dict_idx2bkg, self.dict_bkg2reasoningprocess = load_chem_annotation(args.chem_annotation_path, self.args.if_use_strict_survey_question, self.args.if_use_background_survey)   
        else:
            print("INFO: Using custom_rq and custom_bs.")
            assert custom_rq != None
            self.bkg_q_list = [custom_rq]
            self.dict_bkg2survey = {custom_rq: custom_bs}
            self.dict_idx2bkg = {0: custom_rq}   
        ## Load inspiration corpus (by default is the groundtruth inspiration papers and random high-quality papers)
        # title_abstract_collector: [[title, abstract], ...]
        # dict_title_2_abstract: {'title': 'abstract', ...}
        self.title_abstract_collector, self.dict_title_2_abstract = load_dict_title_2_abstract(title_abstract_collector_path=args.custom_inspiration_corpus_path)
        ## Load the selected inspirations from the inspiration corpus (results from inspiration_screening.py)
        if args.if_use_gdth_insp == 0:
            # organized_insp: {'bq': [[title, reason], [title, reason], ...]}
            self.organized_insp, self.dict_bkg_insp2idx, self.dict_bkg_idx2insp = load_found_inspirations(inspiration_path=args.inspiration_dir, idx_round_of_first_step_insp_screening=args.idx_round_of_first_step_insp_screening)
        else:
            # organized_insp: {'bq': [[title, reason], [title, reason], ...]}
            bkg_q = self.bkg_q_list[args.background_question_id]
            self.organized_insp, self.dict_bkg_insp2idx, self.dict_bkg_idx2insp = load_groundtruth_inspirations_as_screened_inspirations(bkg_q=bkg_q, dict_bkg2insp=self.dict_bkg2insp)
            # change the max_inspiration_search_steps to the number of inspirations in the background question to avoid exception that the number of inspirations is less than the max_inspiration_search_steps
            self.args.max_inspiration_search_steps = len(self.organized_insp[bkg_q])
            print("set self.args.max_inspiration_search_steps to {}".format(len(self.organized_insp[bkg_q])))
            print("Warning: using groundtruth inspirations for hypothesis generation..")
            print("bkg_q: ", bkg_q)


    # hypothesis_collection: {backgroud_question: {core_insp_title: [[hypothesis, reasoning process], ...]}, ...}
    ## Input
    # background_question_id: int
    # inspiration_ids: list; [-1]: iterate over all inspirations; otherwise: only generate hypothesis for specified inspiration ids in the list
    # final_data_collection: previously processed data. If None, we will initialize it, else we will keep its existing inspiration islands, and develop inspirations in input inspiration_ids but not in final_data_collection
    ## Output
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    def hypothesis_generation_for_one_background_question(self, background_question_id, inspiration_ids=[-1], final_data_collection=None):
        ### intra-EA mutation 
        print("\nHypothesis generation for one background question..")
        assert type(inspiration_ids) == list
        # backgroud_question
        backgroud_question = self.dict_idx2bkg[background_question_id]
        # screened_insp_cur_bq: [[title, reason], [title, reason], ...]
        screened_insp_cur_bq = self.organized_insp[backgroud_question]
        # not a sufficient, but a necessary condition to make sure the ids in inspiration_ids correspond to the ids in screened_insp_cur_bq
        assert max(inspiration_ids) < len(screened_insp_cur_bq), "inspiration_ids should be less than the number of inspirations in the background question: max(inspiration_ids): {}; len(screened_insp_cur_bq): {}".format(max(inspiration_ids), len(screened_insp_cur_bq))
        # initialize final_data_collection if it is None
        if final_data_collection == None:
            final_data_collection = {}            
        if backgroud_question not in final_data_collection:
            final_data_collection[backgroud_question] = {}
        # iterate over each core inspiration
        for cur_insp_id in range(len(screened_insp_cur_bq)):
            cur_insp_title = screened_insp_cur_bq[cur_insp_id][0]
            if -1 not in inspiration_ids and cur_insp_id not in inspiration_ids:
                continue
            # only develop hypothesis for the inspirations that are in given inspiration_ids but are not in final_data_collection[backgroud_question]
            if cur_insp_title in final_data_collection[backgroud_question]:
                continue
            print("cur_insp_id: {}; cur_insp_title: {}".format(cur_insp_id, cur_insp_title))
            # generate hypothesis for one background question and one inspiration
            # hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}
            hypthesis_mutation_collection = self.hypothesis_generation_for_one_bkg_one_insp(background_question_id, cur_insp_id)
            # save to final_data_collection
            final_data_collection[backgroud_question][cur_insp_title] = hypthesis_mutation_collection
        
        # save file
        if self.args.if_save:
            self.save_file(final_data_collection, self.args.output_dir)

        # inter-EA recombination and self-explore extra knowledge for the second, the third, ... inspiration exploration step
        if self.args.max_inspiration_search_steps >= 2:
            for cur_step_id in range(2, self.args.max_inspiration_search_steps+1):
                final_data_collection = self.controller_additional_inspiration_step_hypothesis_generation(background_question_id, final_data_collection, step_id=cur_step_id)
                # save file
                if self.args.if_save:
                    self.save_file(final_data_collection, self.args.output_dir)
        return final_data_collection

    
    ## Function
    # Controller for the second inspiration step hypothesis generation (by second round inspiration screening or self-exploration of extra knowledge)
    #   Mainly to determine which recom_inspiration_ids and which self_explore_inspiration_ids for further exploration
    #       1. when recom_inspiration_ids == [] or self_explore_inspiration_ids == [], we will determine them based on the self-evaluation scores of the best hypothesis from each inspiration
    #       2. when -1 in recom_inspiration_ids or -1 in self_explore_inspiration_ids, we will explore over all inspirations
    #       3. when recom_inspiration_ids or self_explore_inspiration_ids is specified, we will explore over the specified inspirations, w/o determining them based on the self-evaluation scores of the best hypothesis from each inspiration
    ## Input
    #   step_id: int; 1: about the first layer of inspiration/hypothesis node; 2: about the second layer of inspiration/hypothesis node; 3: about the third layer of inspiration/hypothesis node; ...
    def controller_additional_inspiration_step_hypothesis_generation(self, background_question_id, final_data_collection, step_id):
        assert step_id >= 2
        if self.args.if_mutate_between_diff_insp == 1 or self.args.if_self_explore == 1:
            ## Obtain a ranked list of core inspiration ids based on the average score of the best hypothesis from each inspiration
            bkg_question = self.dict_idx2bkg[background_question_id]
            # top_hypothesis_collection: {core_insp_title: [[hypothesis, hypothesis_score, [mutation_id], ave_score], ...], ...}; a subset of particular_round_hypothesis_collection; 
            #     step_id == 1: select the best hyp from each insp island; 
            #     step_id >= 2: only collect the top hypothesis nodes (controlled by args.recom_num_beam_size) in an overall ranking of all hypothesis nodes (if a core_insp_title has no hypothesis node in the top nodes, we will abandon this core_insp_title key)
            # ranked_top_core_insp_id_hyp_ave_score_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...], in descending order (higher average_score first); used to determine which inspirations to explore in the next round; a direct translation from top_hypothesis_collection
            # step_id is set to step_id - 1, since we want to find the top ranked (insp, hyp) pair from the previous layer/step
            # we keep all data here without filtering (top_ratio_to_keep = 1.0); we will filter them inside inter-EA or self-explore functions
            print("\n\nRanking the hypothesis nodes in step {} to select top ones for further exploitation..".format(step_id-1))
            top_hypothesis_collection, ranked_top_core_insp_id_hyp_ave_score_list = self.select_top_self_evaluated_hypothesis(final_data_collection, bkg_question, step_id-1, top_ratio_to_keep=1.0)
            # display the ranked core inspirations
            for idx in range(len(ranked_top_core_insp_id_hyp_ave_score_list)):
                print("\trank: {}, ave_score: {:.2f}; insp_id: {}; insp_title: {}".format(idx+1, ranked_top_core_insp_id_hyp_ave_score_list[idx][4], ranked_top_core_insp_id_hyp_ave_score_list[idx][0], self.dict_bkg_idx2insp[bkg_question][ranked_top_core_insp_id_hyp_ave_score_list[idx][0]]))
            print("Number of (all) ranked hypothesis nodes to determine which node to further exploit: {}".format(len(ranked_top_core_insp_id_hyp_ave_score_list)))

        ### inter-EA recombination
        if self.args.if_mutate_between_diff_insp == 1:
            final_data_collection = hyp_gene_ea.recombinational_mutation_between_diff_insp(background_question_id=background_question_id, ranked_top_insp_list=ranked_top_core_insp_id_hyp_ave_score_list, recom_inspiration_ids_user_input=self.args.recom_inspiration_ids, final_data_collection=final_data_collection, step_id=step_id)

        ### self-explore extra knowledge
        if self.args.if_self_explore == 1:
            final_data_collection = hyp_gene_ea.self_explore_extra_knowledge_one_bkg_multiple_insp_node(background_question_id=background_question_id, ranked_top_insp_list=ranked_top_core_insp_id_hyp_ave_score_list, self_explore_inspiration_ids_user_input=self.args.self_explore_inspiration_ids, final_data_collection=final_data_collection, step_id=step_id)
        
        return final_data_collection
    



    ## Function
    # re-combinational mutation between the final hypothesis developed from the same bkq and different insp
    ## Input
    # background_question_id: int
    # ranked_top_insp_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...], in descending order (higher average_score first); used to determine which inspirations to explore in the next step
    # recom_inspiration_ids: list; [0, 1, 2, ...]: the inspiration ids in the background question; [-1]: iterate over all inspirations; identify the inspiration ids that we want to conduct inter-EA recombination
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom'
    # step_id: int; 1: find the first inspiration; 2: find the second inspiration; 3: find the third inspiration; ...
    ## Output
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom' & 'inter_recom'
    #     hypthesis_mutation_collection['inter_com(_{step})']: {core_insp_title_best_mutation_id: {matched_insp_title0: [[hyp0, reasoning process0, feedback0], ...], ...}}
    def recombinational_mutation_between_diff_insp(self, background_question_id, ranked_top_insp_list, recom_inspiration_ids_user_input, final_data_collection, step_id):
        # this function is at least used for the second layer of nodes
        assert step_id >= 2
        # this_recom_mutation_id = "inter_recom" if step_id == 2 else "inter_recom_{}".format(step_id-1)
        this_recom_mutation_id = "inter_recom_{}".format(step_id-1)
        print("\n\nInter-EA Step: {}".format(step_id))
        ## get filtered_ranked_top_insp_list (filter based on recom_inspiration_ids_user_input or args.recom_num_beam_size)
        # filtered_ranked_top_insp_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...]; used to determine which nodes to further expand / explore
        if -1 in recom_inspiration_ids_user_input:
            # no filter, keep all
            assert len(recom_inspiration_ids_user_input) == 1
            filtered_ranked_top_insp_list = ranked_top_insp_list
        elif len(recom_inspiration_ids_user_input) > 0:
            # filter twice; first is by recom_inspiration_ids_user_input; second is by recom_num_beam_size
            filtered_ranked_top_insp_list = [item for item in ranked_top_insp_list if item[0] in recom_inspiration_ids_user_input]
            num_top_insp_to_keep = min(self.args.recom_num_beam_size, len(filtered_ranked_top_insp_list))
            # smallest_ave_score_threshold: to reduce randomness caused by the random order of same scored hypothesis nodes
            smallest_ave_score_threshold = filtered_ranked_top_insp_list[num_top_insp_to_keep-1][4]
            filtered_ranked_top_insp_list = [item for item in filtered_ranked_top_insp_list if item[4] >= smallest_ave_score_threshold]
        else:
            # filter once; by recom_num_beam_size
            num_top_insp_to_keep = min(self.args.recom_num_beam_size, len(ranked_top_insp_list))
            # smallest_ave_score_threshold: to reduce randomness caused by the random order of same scored hypothesis nodes
            smallest_ave_score_threshold = ranked_top_insp_list[num_top_insp_to_keep-1][4]
            filtered_ranked_top_insp_list = [item for item in ranked_top_insp_list if item[4] >= smallest_ave_score_threshold]
        print("Number of selected hypothesis nodes to further exploit: ", len(filtered_ranked_top_insp_list))

        ## Prepare background and inspiration information
        backgroud_question = self.dict_idx2bkg[background_question_id]
        # backgroud_survey
        backgroud_survey = self.dict_bkg2survey[backgroud_question]
        # screened_insp_cur_bq: [[title, reason], [title, reason], ...]
        screened_insp_cur_bq = self.organized_insp[backgroud_question]

        # select the best hypothesis from each inspiration (with highest self-evaluation scores)
        # best_hypothesis_collection_for_recomb: {core_insp_title: [[best_hypothesis, best_hypothesis_score, [best_mutation_id], best_ave_score]], ...}
        best_hypothesis_collection_for_recomb, _ = self.select_top_self_evaluated_hypothesis(final_data_collection, backgroud_question, step_id=1, top_ratio_to_keep=1.0)
        # iterate over each (insp, hyp) pair in filtered_ranked_top_insp_list
        for cur_node_id in range(len(filtered_ranked_top_insp_list)):
            cur_insp_id = filtered_ranked_top_insp_list[cur_node_id][0]
            # add abstract to screened_insp_cur_bq in addition to title and reason
            cur_insp_core_node = screened_insp_cur_bq[cur_insp_id]
            cur_abstract = get_item_from_dict_with_very_similar_but_not_exact_key(self.dict_title_2_abstract, cur_insp_core_node[0])
            cur_insp_core_node.append(cur_abstract)
            # cur_insp_title
            cur_insp_title = cur_insp_core_node[0]
            print("\nInter-EA recombination for cur_insp_id: {}; cur_insp_title: {}".format(cur_insp_id, cur_insp_title))
            ### recombinational mutation
            ## first select the best hypothesis from every other inspiration (with highest average self-evaluation score)
            # other_mutations: [[insp_title0, insp_abstract0, hyp0], [insp_title1, insp_abstract1, hyp1], ...], here 0, 1 indicates the id of differnt inspirations
            # get cur_node_search_trail (all previous mutation ids) to avoid select the same inspiration again: ['mut_id_0', 'mut_id_1', ...]
            cur_node_search_trail_raw = filtered_ranked_top_insp_list[cur_node_id][3]
            cur_node_search_trail = [cur_insp_title]
            for cur_node_search_trail_item in cur_node_search_trail_raw:
                if ";" in cur_node_search_trail_item:
                    cur_node_search_trail += cur_node_search_trail_item.split(";")
                else:
                    cur_node_search_trail.append(cur_node_search_trail_item)
            other_mutations = [[tmp_insp_title, get_item_from_dict_with_very_similar_but_not_exact_key(self.dict_title_2_abstract, tmp_insp_title), best_hypothesis_collection_for_recomb[tmp_insp_title][0][0]] for tmp_insp_title in best_hypothesis_collection_for_recomb if tmp_insp_title not in cur_node_search_trail]
            # this_mutation: hypothesis developed from the current inspiration; text
            this_mutation = filtered_ranked_top_insp_list[cur_node_id][1]
            ## screening all other (inspiration, hypothesis) pairs for recombination with the current (inspiration, hypothesis) pair
            print("Inspiration screening..")
            assert len(other_mutations) >= 1
            # each screening inference will select 3 inspirations
            if len(other_mutations) <= self.args.num_screening_keep_size:
                selected_other_mutations = other_mutations
            elif len(other_mutations) <= self.args.num_screening_window_size:
                # selected_other_mutations: a subset of other_mutations; [[insp_title0, insp_abstract0, hyp0], [insp_title1, insp_abstract1, hyp1], ...]
                selected_other_mutations = self.additional_round_inspiration_screening(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=other_mutations, this_mutation=this_mutation)
            else:
                num_screening_itr = math.ceil(len(other_mutations) / self.args.num_screening_window_size)
                selected_other_mutations = []
                for cur_itr in range(num_screening_itr):
                    cur_max_other_mutations_id = min((cur_itr+1)*self.args.num_screening_window_size, len(other_mutations))
                    cur_other_mutations = other_mutations[cur_itr*self.args.num_screening_window_size: cur_max_other_mutations_id]
                    if len(cur_other_mutations) <= self.args.num_screening_keep_size:
                        cur_selected_other_mutations = cur_other_mutations
                    else:
                        cur_selected_other_mutations = self.additional_round_inspiration_screening(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=cur_other_mutations, this_mutation=this_mutation)    
                    selected_other_mutations += cur_selected_other_mutations
                    print("\tlen(cur_selected_other_mutations): {}".format(len(cur_selected_other_mutations)))
                    if len(cur_selected_other_mutations) == 0:
                        print("Warning: len(cur_selected_other_mutations) == 0; {}".format(cur_selected_other_mutations))
            selected_other_mutations_titles = [item[0] for item in selected_other_mutations]
            print("\tSelected {} inspirations for additional_round_inspiration_screening: {}".format(len(selected_other_mutations), selected_other_mutations_titles))
            ## recombinational mutation between the current (inspiration, hypothesis) pair and the most matched (inspiration, hypothesis) pair
            for cur_other_mutation in selected_other_mutations:
                # print("\tWorking on cur_other_mutation: {}...".format(cur_other_mutation[0]))
                # cur_other_mutation: [insp_title, insp_abstract, hyp]
                # cur_hypothesis_collection: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]
                cur_hypothesis_collection = self.hyothesis_generation_with_refinement(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=cur_other_mutation, recombination_type=2, this_mutation=this_mutation)
                if this_recom_mutation_id not in final_data_collection[backgroud_question][cur_insp_title]:
                    final_data_collection[backgroud_question][cur_insp_title][this_recom_mutation_id] = {}
                # only pick the lask mutation id still might cause misundertanding, so we join all past mutation ids as the current mutation id
                cur_node_prev_round_branch_mutation_id = ";".join(filtered_ranked_top_insp_list[cur_node_id][3])
                if cur_node_prev_round_branch_mutation_id not in final_data_collection[backgroud_question][cur_insp_title][this_recom_mutation_id]:
                    final_data_collection[backgroud_question][cur_insp_title][this_recom_mutation_id][cur_node_prev_round_branch_mutation_id] = {}
                # print scores
                print("\tcur_insp: {}; hyp_numerical_self_eval: {}".format(cur_other_mutation[0], cur_hypothesis_collection[-1][-1][0]))
                final_data_collection[backgroud_question][cur_insp_title][this_recom_mutation_id][cur_node_prev_round_branch_mutation_id][cur_other_mutation[0]] = cur_hypothesis_collection
        return final_data_collection
    

    

    ## Function
    # screening all other (inspiration, hypothesis) pairs for recombination with the current (inspiration, hypothesis) pair
    ## Input
    # backgroud_question: text
    # backgroud_survey: text
    # cur_insp_core_node: [title, reason, abstract]
    # other_mutations: [[insp_title0, insp_abstract0, hyp0], [insp_title1, insp_abstract1, hyp1], ...]
    # this_mutation: text of a hypothesis (already developed based on cur_insp_core_node)
    ## Output
    # selected_other_mutations: a subset of other_mutations; [[insp_title0, insp_abstract0, hyp0], [insp_title1, insp_abstract1, hyp1], ...]
    def additional_round_inspiration_screening(self, backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations, this_mutation):
        # prompts
        prompts = instruction_prompts("additional_round_inspiration_screening")
        assert len(prompts) == 6
        # cur_insp_core_node_prompt
        cur_insp_core_node_prompt = "Title: {}; Abstract: {}.".format(cur_insp_core_node[0], cur_insp_core_node[2])
        # other_mutations_prompt
        other_mutations_prompt = ""
        for cur_other_mutation_id, cur_other_mutation in enumerate(other_mutations):
            other_mutations_prompt += "Next we will introduce potential inspiration candidate {}. Title: {}; Abstract: {}. This inspiration has been leveraged to generate hypothesis for the given background question. The hypothesis is: {}. \n".format(cur_other_mutation_id, cur_other_mutation[0], cur_other_mutation[1], cur_other_mutation[2])
        full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + this_mutation + prompts[4] + other_mutations_prompt + prompts[5]
        # generation
        # structured_extra_knowledge: [[Title0, Reason0], [Title1, Reason1], ...]
        # we might want the temperature for inspiration retrieval to be zero, for better reflecting heuristics & stable performance
        structured_extra_knowledge = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Title:', 'Reason:'], temperature=0.0, restructure_output_model_name=self.args.model_name)
        # structured_extra_knowledge = exchange_order_in_list(structured_extra_knowledge)
        structured_extra_knowledge = [[recover_generated_title_to_exact_version_of_title(list(self.dict_title_2_abstract.keys()), item[0]), item[1]] for item in structured_extra_knowledge]
        # selected_titles: [Title0, Title1, ...]
        selected_titles = [item[0] for item in structured_extra_knowledge]
        selected_other_mutations = [cur_other_mutation for cur_other_mutation in other_mutations if cur_other_mutation[0] in selected_titles]
        return selected_other_mutations



    ## Function
    # select the best hypothesis from each inspiration (with highest self-evaluation scores) (with one backgroud_question root node)
    # select the nodes from the first inspiration step
    ## Input
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom'
    #     hypthesis_mutation_collection['inter_com']: {core_insp_title_best_mutation_id: {matched_insp_title0: [[hyp0, reasoning process0, feedback0], ...], ...}}
    # backgroud_question: text
    # step_id: int; 1: find the first inspiration; 2: find the second inspiration; 3: find the third inspiration; ...
    # top_ratio_to_keep: determines how many hypotheses to keep to enter the next step
    ## Output
    #   top_hypothesis_collection: {core_insp_title: [[hypothesis, hypothesis_score, [mutation_id], ave_score], ...], ...}; a subset of particular_round_hypothesis_collection; 
    #       step_id == 1: select the best hyp from each insp island; 
    #       step_id >= 2: only collect the top hypothesis nodes (controlled by args.recom_num_beam_size) in an overall ranking of all hypothesis nodes (if a core_insp_title has no hypothesis node in the top nodes, we will abandon this core_insp_title key)
    #   ranked_core_insp_id_ave_score_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...]: a direct translation from top_hypothesis_collection; used to determine which inspirations to explore in the next round; 
    def select_top_self_evaluated_hypothesis(self, final_data_collection, backgroud_question, step_id=1, top_ratio_to_keep=1.0):
        assert type(step_id) == int and step_id >= 1
        ## get particular_round_hypothesis_collection
        # particular_round_hypothesis_collection: {core_insp_title: [[hypothesis, hypothesis_score, [mutation_id], ave_score], ...], ...}; [mutation_id]: [mut_id_0, mut_id_1, ...]; the list of list is ranked by ave_score in descending order; collect all hypothesis node at that particular round
        particular_round_hypothesis_collection = {}
        for cur_insp_title in final_data_collection[backgroud_question]:
            particular_round_hypothesis_collection[cur_insp_title] = []
            for cur_mutation_id in final_data_collection[backgroud_question][cur_insp_title]:
                if step_id == 1:
                    if type(cur_mutation_id) != str:
                        print("cur_mutation_id: ", cur_mutation_id)
                    if "inter_recom" in cur_mutation_id or "self_explore" in cur_mutation_id:
                        continue    
                    # cur_hypothesis_score: [valid_score, novel_score, significance_score, potential_score]
                    cur_hypothesis_score = final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id][-1][3][0]
                    assert len(cur_hypothesis_score) == 4
                    cur_ave_score = sum(cur_hypothesis_score) / len(cur_hypothesis_score)
                    cur_hyp = final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id][-1][0]
                    particular_round_hypothesis_collection[cur_insp_title].append([cur_hyp, cur_hypothesis_score, [cur_mutation_id], cur_ave_score])
                else:
                    # cur_focus_mutation_id = "inter_recom" if step_id == 2 else "inter_recom_{}".format(step_id-1)
                    cur_focus_mutation_id = "inter_recom_{}".format(step_id-1)
                    if cur_mutation_id != cur_focus_mutation_id:
                        continue
                    for cur_prev_round_mut_id in final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id]:
                        for cur_cur_round_mut_id in final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id][cur_prev_round_mut_id]:
                            cur_hypothesis_score = final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id][cur_prev_round_mut_id][cur_cur_round_mut_id][-1][3][0]
                            assert len(cur_hypothesis_score) == 4
                            cur_ave_score = sum(cur_hypothesis_score) / len(cur_hypothesis_score)
                            cur_hyp = final_data_collection[backgroud_question][cur_insp_title][cur_mutation_id][cur_prev_round_mut_id][cur_cur_round_mut_id][-1][0]
                            particular_round_hypothesis_collection[cur_insp_title].append([cur_hyp, cur_hypothesis_score, [cur_prev_round_mut_id, cur_cur_round_mut_id, cur_mutation_id], cur_ave_score])
        # sort particular_round_hypothesis_collection
        for cur_insp_title in particular_round_hypothesis_collection:
            particular_round_hypothesis_collection[cur_insp_title] = sorted(particular_round_hypothesis_collection[cur_insp_title], key=lambda x: x[3], reverse=True)

        ## get top_hypothesis_collection (same template with particular_round_hypothesis_collection)
        #   top_hypothesis_collection: {core_insp_title: [[hypothesis, hypothesis_score, [mutation_id], ave_score], ...], ...}; a subset of particular_round_hypothesis_collection; 
        #   only collect the top hypothesis nodes in an overall ranking of all hypothesis nodes (if a core_insp_title has no hypothesis node in the top nodes, we will abandon this core_insp_title key)
        top_hypothesis_collection = {}
        if step_id == 1:
            # when step_id == 1: we collect the best hyp from each insp island as the all_hypothesis_collection
            all_hypothesis_collection = [[cur_insp_title, particular_round_hypothesis_collection[cur_insp_title][0]] for cur_insp_title in particular_round_hypothesis_collection]
        else:
            # when step_id >= 2: we collect all hyp from each insp island as the all_hypothesis_collection
            all_hypothesis_collection = [[cur_insp_title, item] for cur_insp_title in particular_round_hypothesis_collection for item in particular_round_hypothesis_collection[cur_insp_title]]
        all_hypothesis_collection = sorted(all_hypothesis_collection, key=lambda x: x[1][3], reverse=True)
        num_top_hypothesis_nodes = max(1, int(len(all_hypothesis_collection)*top_ratio_to_keep))
        # cur_hypothesis_info: [hypothesis, hypothesis_score, [mutation_id], ave_score]
        for cur_insp_title, cur_hypothesis_info in all_hypothesis_collection[:num_top_hypothesis_nodes]:
            if cur_insp_title not in top_hypothesis_collection:
                top_hypothesis_collection[cur_insp_title] = []
            top_hypothesis_collection[cur_insp_title].append(cur_hypothesis_info)

        ## get ranked_core_insp_id_ave_score_list: a direct transfer of top_hypothesis_collection: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...]
        ranked_core_insp_id_ave_score_list = []
        for cur_insp_title in top_hypothesis_collection:
            for cur_hypothesis_info in top_hypothesis_collection[cur_insp_title]:
                ranked_core_insp_id_ave_score_list.append([self.dict_bkg_insp2idx[backgroud_question][cur_insp_title], cur_hypothesis_info[0], cur_hypothesis_info[1], cur_hypothesis_info[2], cur_hypothesis_info[3]])
        ranked_core_insp_id_ave_score_list = sorted(ranked_core_insp_id_ave_score_list, key=lambda x: x[4], reverse=True)

        return top_hypothesis_collection, ranked_core_insp_id_ave_score_list



    ## Input
    # background_question_id: int
    # ranked_top_insp_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...], in descending order (higher average_score first); used to determine which inspirations to explore in the next step
    # self_explore_inspiration_ids: list; [0, 1, 2, ...]: the inspiration ids in the background question; [-1]: iterate over all inspirations; identify the inspiration ids that we want to conduct inter-EA recombination
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], ..., [hypn, reasoning processn, feedbackn, [[valid_score, novel_score, significance_score, potential_score], [reason0, reason1, reason2, reason3]]]]}; mutation_id: 0, 1, 2, ... & 'recom' (& 'inter_recom')
    ## Output
    # final_data_collection: {backgroud_question: {core_insp_title: hypthesis_mutation_collection, ...}, ...}
    #     hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}; mutation_id: 0, 1, 2, ... & 'recom' (& 'inter_recom') & 'self_explore'
    def self_explore_extra_knowledge_one_bkg_multiple_insp_node(self, background_question_id, ranked_top_insp_list, self_explore_inspiration_ids_user_input, final_data_collection, step_id):
        assert step_id >= 2
        this_explore_mutation_id = "self_explore" if step_id == 2 else "self_explore_{}".format(step_id-1)
        print("\n\nSelf-explore step: {}".format(step_id))
        ## get filtered_ranked_top_insp_list (filter based on recom_inspiration_ids_user_input or args.self_explore_num_beam_size)
        # filtered_ranked_top_insp_list: [[core_insp_id, hypothesis, hypothesis_score, [mutation_id], ave_score], ...]
        if -1 in self_explore_inspiration_ids_user_input:
             # no filter, keep all
            assert len(self_explore_inspiration_ids_user_input) == 1
            filtered_ranked_top_insp_list = ranked_top_insp_list
        elif len(self_explore_inspiration_ids_user_input) > 0:
            # filter twice; first is by self_explore_inspiration_ids_user_input; second is by self_explore_num_beam_size
            filtered_ranked_top_insp_list = [item for item in ranked_top_insp_list if item[0] in self_explore_inspiration_ids_user_input]
            num_top_insp_to_keep = self.args.self_explore_num_beam_size
            smallest_ave_score_threshold = filtered_ranked_top_insp_list[num_top_insp_to_keep-1][4]
            filtered_ranked_top_insp_list = [item for item in filtered_ranked_top_insp_list if item[4] >= smallest_ave_score_threshold]
        else:
            # filter once; by self_explore_num_beam_size
            num_top_insp_to_keep = self.args.self_explore_num_beam_size
            smallest_ave_score_threshold = ranked_top_insp_list[num_top_insp_to_keep-1][4]
            filtered_ranked_top_insp_list = [item for item in ranked_top_insp_list if item[4] >= smallest_ave_score_threshold]
        print("Number of selected hypothesis nodes to further exploit: ", len(filtered_ranked_top_insp_list))

        ## Prepare background and inspiration information
        backgroud_question = self.dict_idx2bkg[background_question_id]
        # backgroud_survey
        backgroud_survey = self.dict_bkg2survey[backgroud_question]
        # screened_insp_cur_bq: [[title, reason], [title, reason], ...]
        screened_insp_cur_bq = self.organized_insp[backgroud_question]

        # for cur_insp_id in range(len(screened_insp_cur_bq)):
        for cur_node_id in range(len(filtered_ranked_top_insp_list)):
            cur_insp_id = filtered_ranked_top_insp_list[cur_node_id][0]
            # add abstract in addition to title and reason in screened_insp_cur_bq
            cur_insp_core_node = screened_insp_cur_bq[cur_insp_id]
            cur_abstract = get_item_from_dict_with_very_similar_but_not_exact_key(self.dict_title_2_abstract, cur_insp_core_node[0])
            cur_insp_core_node.append(cur_abstract)
            # cur_insp_title
            cur_insp_title = cur_insp_core_node[0]
            print("cur_insp_id: {}; cur_insp_title: {}".format(cur_insp_id, cur_insp_title))
            # self-explore extra knowledge for one background question and one inspiration and one hypothesis
            cur_hypothesis = filtered_ranked_top_insp_list[cur_node_id][1]
            cur_prev_mutation_ids = filtered_ranked_top_insp_list[cur_node_id][3]
            cur_prev_mutation_ids = ";".join(cur_prev_mutation_ids)
            # self_explored_knowledge_hypothesis_collection: {mutation_id: [[extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0], ...], ...}
            self_explored_knowledge_hypothesis_collection = self.self_explore_extra_knowledge_one_bkg_one_insp_node_full_steps(backgroud_question, backgroud_survey, cur_insp_core_node, cur_hypothesis)
            # save to final_data_collection
            if this_explore_mutation_id not in final_data_collection[backgroud_question][cur_insp_title]:
                final_data_collection[backgroud_question][cur_insp_title][this_explore_mutation_id] = {cur_prev_mutation_ids: self_explored_knowledge_hypothesis_collection}
            else:
                if cur_prev_mutation_ids in final_data_collection[backgroud_question][cur_insp_title][this_explore_mutation_id]:
                    print("Warning: cur_prev_mutation_ids: {} already exists in final_data_collection[{}][{}][{}]".format(cur_prev_mutation_ids, backgroud_question, cur_insp_title, this_explore_mutation_id))
                final_data_collection[backgroud_question][cur_insp_title][this_explore_mutation_id][cur_prev_mutation_ids] = self_explored_knowledge_hypothesis_collection
        return final_data_collection




    ## Function
    # generate hypotheses (w/ or w/o mutation) (w/ or w/o refinement) for one background question and one inspiration
    ## Input
    # background_question_id: int
    # inspiration_id: int
    ## Output
    # hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}
    def hypothesis_generation_for_one_bkg_one_insp(self, background_question_id, inspiration_id):
        assert self.args.if_mutate_inside_same_bkg_insp in [0, 1]
        ## prepare background and inspiration information
        # backgroud_question
        backgroud_question = self.dict_idx2bkg[background_question_id]
        # backgroud_survey
        backgroud_survey = self.dict_bkg2survey[backgroud_question]
        # screened_insp_cur_bq: [[title, reason], [title, reason], ...]
        screened_insp_cur_bq = self.organized_insp[backgroud_question]
        # add abstract in addition to title and reason in screened_insp_cur_bq
        cur_insp_core_node = screened_insp_cur_bq[inspiration_id]
        cur_title = cur_insp_core_node[0]
        # cur_abstract = self.dict_title_2_abstract[cur_title]
        cur_abstract = get_item_from_dict_with_very_similar_but_not_exact_key(self.dict_title_2_abstract, cur_title)
        # cur_insp_core_node: [title, reason, abstract]
        cur_insp_core_node.append(cur_abstract)

        ## generate several distinct mutation hyp, and develop them by refinement for each line of mutation
        # hypthesis_mutation_collection: {mutation_id: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]}
        hypthesis_mutation_collection = {}
        # hypothesis_collection: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]
        hypothesis_collection = self.hyothesis_generation_with_refinement(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=None)
        hypthesis_mutation_collection['0'] = hypothesis_collection
        if self.args.if_mutate_inside_same_bkg_insp == 1:
            for cur_mutation_id in range(1, self.args.num_mutations):
                # other_mutations: the most refined hypothesis from other mutations
                other_mutations = [hypthesis_mutation_collection[mut_id][-1][0] for mut_id in hypthesis_mutation_collection]
                hypothesis_collection = self.hyothesis_generation_with_refinement(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=other_mutations)
                hypthesis_mutation_collection[str(cur_mutation_id)] = hypothesis_collection

        ## re-combinational mutation between different mutation lines developed from the same bkq and insp
        if self.args.if_mutate_inside_same_bkg_insp == 1:
            print("Recombinational mutation")
            assert len(hypthesis_mutation_collection) > 1
            other_mutations = [hypthesis_mutation_collection[mut_id][-1][0] for mut_id in hypthesis_mutation_collection]
            hypothesis_collection = self.hyothesis_generation_with_refinement(backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=other_mutations, recombination_type=1)
            hypthesis_mutation_collection['recom'] = hypothesis_collection
        return hypthesis_mutation_collection
            

    ## Function
    # generate hypothesis with refinement (w/ or w/o mutation)
    ## Input:
    # backgroud_question: text  
    # backgroud_survey: text
    # cur_insp_core_node: [title, reason, abstract]
    # other_mutations: [hyp0, hyp1, ...], here 0, 1 indicates the id of mutation; or None; used for developing a new mutation line
    # recombination_type: 0/1/2; 
    #   2: recombinational mutation between the final hypothesis developed from (the same background and) different inspirations;
    #   1: recombinational mutation between the last refined hypothesis over each mutation line. Each line is developed from the same background and the same inspiration; 
    #   0: develop a new mutation line
    # this_mutation: text of a hypothesis (already developed based on cur_insp_core_node); only used when recombination_type=2 (recombinational mutation between the final hypothesis developed from the same bkq and different insp, and this_mutation is the hypothesis developed from cur_insp_core_node)
    # if_self_eval_for_final_hyp: bool, default is True; whether we provide numerical evaluation for the final hypothesis (final hypothesis: the last refined hypothesis)
    ## Output:
    # hypothesis_collection: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]
    def hyothesis_generation_with_refinement(self, backgroud_question, backgroud_survey, cur_insp_core_node, other_mutations=None, recombination_type=0, this_mutation=None, if_self_eval_for_final_hyp=True):
        assert recombination_type in [0, 1, 2]
        # this_mutation is used iff recombination_type=2
        assert this_mutation == None if recombination_type != 2 else this_mutation != None
        print("New mutation line is developing..")
        # hypothesis_collection: [[hyp0, reasoning process0, feedback0], [hyp1, reasoning process1, feedback1], ...]
        hypothesis_collection = []
        for cur_refine_iter in range(self.args.num_itr_self_refine):
            # set parameters: same_mutation_prev_hyp, hyp_feedback, and other_mutations
            if cur_refine_iter == 0:
                same_mutation_prev_hyp, hyp_feedback = None, None
            else:
                # when recombine hyp from different insp islands, we need to keep seeing the different hyps from different insp islands during refinement
                if recombination_type == 1 or recombination_type == 2:
                    same_mutation_prev_hyp, hyp_feedback = hypothesis_collection[-1][0], hypothesis_collection[-1][2]
                elif recombination_type == 0:
                    same_mutation_prev_hyp, hyp_feedback = hypothesis_collection[-1][0], hypothesis_collection[-1][2]
                    # when developing the second/third/... mutation line, and it is not the first hypothesis in this mutation line, it is not necessary to attend to hypotheses in other mutation lines (other_mutations)
                    other_mutations = None
                else:
                    raise NotImplementedError
            # set parameters: if_with_external_knowledge_feedback 
            #   (only during the first iteration, the refinement will consider to add external knowledge to stick bkg and insp)
            #   it is to prevent too much additional information in the final hypothesis (previously we add external knowledge at every refinement step)
            if self.args.if_consider_external_knowledge_feedback_during_second_refinement and cur_refine_iter == 1:
                if_with_external_knowledge_feedback = True
            else:
                if_with_external_knowledge_feedback = False
            # cur_hypothesis_and_reasoning_process: [hyp, reasoning process]
            cur_hypothesis_and_reasoning_process = self.one_inference_for_one_hyp_gene(backgroud_question, backgroud_survey, cur_insp_core_node, same_mutation_prev_hyp=same_mutation_prev_hyp, hyp_feedback=hyp_feedback, other_mutations=other_mutations, recombination_type=recombination_type, this_mutation=this_mutation)
            # provide feedback
            hyp_feedback = self.hypothesis_refinement(cur_hypothesis_and_reasoning_process, if_with_external_knowledge_feedback=if_with_external_knowledge_feedback)
            # cur_hypothesis_and_reasoning_process: [hyp, reasoning process, feedback]
            cur_hypothesis_and_reasoning_process.append(hyp_feedback)
            # provide numerical evaluation for the final hypothesis
            if cur_refine_iter == self.args.num_itr_self_refine - 1:
                if if_self_eval_for_final_hyp:
                    # provide numerical evaluation for the final hypothesis
                    # hyp_numerical_self_eval: [score_collection, score_reason_collection]
                    # score_collection: ['score0', 'score1', 'score2', 'score3']
                    # score_reason_collection: ['reason0', 'reason1', 'reason2', 'reason3']
                    hyp_numerical_self_eval = self.hypothesis_evaluation(cur_hypothesis_and_reasoning_process)
                    # cur_hypothesis_and_reasoning_process: [hyp, reasoning process, feedback, [score_collection, score_reason_collection]]
                    cur_hypothesis_and_reasoning_process.append(hyp_numerical_self_eval)
            hypothesis_collection.append(cur_hypothesis_and_reasoning_process)
        return hypothesis_collection



    ## Function
    # a full knowledge discovery procedure (to find the second and the third key points) by self-explored extra knowledge for one bkg one insp node (the insp node is the first key point)
    ## Input
    # backgroud_question: text  
    # backgroud_survey: text
    # cur_insp_core_node: [title, reason, abstract]
    # origin_hyp_node: text: the (final) hypothesis from the bkg root node + one insp secondary node
    ## Output
    # self_explored_knowledge_hypothesis_collection: {mutation_id: [[extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0], ...], ...}
    def self_explore_extra_knowledge_one_bkg_one_insp_node_full_steps(self, backgroud_question, backgroud_survey, cur_insp_core_node, origin_hyp_node):
        ## develop one mutation line first, and then develop the second mutation line, and then ...
        self_explored_knowledge_hypothesis_collection = {}
        for cur_mutation_id in range(self.args.num_mutations):
            for cur_iter_explore_id in range(self.args.num_self_explore_steps_each_line):
                if cur_mutation_id == 0 and cur_iter_explore_id == 0:
                    input_hyp = origin_hyp_node
                    other_mutations = None
                else:
                    if cur_iter_explore_id == 0:
                        input_hyp = origin_hyp_node
                    else:
                        input_hyp = self_explored_knowledge_hypothesis_collection[cur_mutation_id][-1][4]
                    if cur_mutation_id == 0:
                        other_mutations = None
                    else:
                        other_mutations = [self_explored_knowledge_hypothesis_collection[mut_id][-1][4] for mut_id in self_explored_knowledge_hypothesis_collection]
                # hypothesis_collection: [extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0(, [score_collection, score_reason_collection])]; "(, [score_collection, score_reason_collection]) added if it is the last hypothesis in each line of mutation"
                if_hyp_need_extra_knowledge, hypothesis_collection = self.self_explore_extra_knowledge_and_hyp_gene_and_refinement_single_step(backgroud_question, backgroud_survey, cur_insp_core_node, input_hyp=input_hyp, other_mutations=other_mutations)
                assert if_hyp_need_extra_knowledge == 'Yes' or if_hyp_need_extra_knowledge == 'No'
                if cur_mutation_id not in self_explored_knowledge_hypothesis_collection:
                    self_explored_knowledge_hypothesis_collection[cur_mutation_id] = []
                # self evaluate at the last hypothesis in each line of mutation
                if cur_iter_explore_id == self.args.num_self_explore_steps_each_line - 1 or if_hyp_need_extra_knowledge == 'No':
                    # provide numerical evaluation for the final hypothesis
                    # hyp_numerical_self_eval: [score_collection, score_reason_collection]
                    # score_collection: ['score0', 'score1', 'score2', 'score3']
                    # score_reason_collection: ['reason0', 'reason1', 'reason2', 'reason3']
                    hyp_numerical_self_eval = self.hypothesis_evaluation([hypothesis_collection[4]])
                    # hypothesis_collection: [extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0, refined_reasoning_process_0, [score_collection, score_reason_collection]]
                    hypothesis_collection.append(hyp_numerical_self_eval)
                    print("\tcur_mutation_id: {}; hyp_numerical_self_eval: {}".format(cur_mutation_id, hyp_numerical_self_eval[0]))
                self_explored_knowledge_hypothesis_collection[cur_mutation_id].append(hypothesis_collection)
                if if_hyp_need_extra_knowledge == 'No':
                    print("No need for extra knowledge, break the loop. cur_mutation_id: {}; cur_iter_explore_id: {}".format(cur_mutation_id, cur_iter_explore_id))
                    break
        return self_explored_knowledge_hypothesis_collection




    ## Function
    # Explore [extra knowledge (once) and hypothesis generation (once) and refinement (once)] (multiple times, controled by hyper-parameter)
    ## Input
    # backgroud_question: text  
    # backgroud_survey: text
    # cur_insp_core_node: [title, reason, abstract]
    # input_hyp: text
    # other_mutations: [hyp0, hyp1, ...], here 0, 1 indicates the id of mutation
    ## Output
    # hypothesis_collection: [extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0]
    def self_explore_extra_knowledge_and_hyp_gene_and_refinement_single_step(self, backgroud_question, backgroud_survey, cur_insp_core_node, input_hyp, other_mutations=None):
        # core insp prompt
        cur_insp_core_node_prompt = "title: {}; abstract: {}.".format(cur_insp_core_node[0], cur_insp_core_node[2])

        ## self-explore of extra knowledge as additional complement key point
        if other_mutations == None:
            prompts = instruction_prompts("self_extra_knowledge_exploration")
            assert len(prompts) == 5
            full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + input_hyp + prompts[4]
        else:
            prompts = instruction_prompts("self_extra_knowledge_exploration_with_other_mutations")
            assert len(prompts) == 6
            other_mutations_prompt = ""
            for cur_other_mutation_id, cur_other_mutation in enumerate(other_mutations):
                cur_other_mutation_prompt = "Next is afterwards hypothesis {} that we want to avoid: {}.\n".format(cur_other_mutation_id, cur_other_mutation)
                other_mutations_prompt += cur_other_mutation_prompt
            full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + input_hyp + prompts[4] + other_mutations_prompt + prompts[5]
        # structured_extra_knowledge: [Yes/No, extra_knowledge/reason for it is complete]
        structured_extra_knowledge = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['If need extra knowledge:', 'Details:'], gene_format_constraint=[0, ['Yes', 'No']], if_only_return_one_structured_gene_component=True, restructure_output_model_name=self.args.model_name)
        if structured_extra_knowledge[0] == 'No':
            hypothesis_collection = [structured_extra_knowledge[1], None, None, None, None]
            return structured_extra_knowledge[0], hypothesis_collection
        ## hypothesis generation
        prompts = instruction_prompts("hypothesis_generation_with_extra_knowledge")
        assert len(prompts) == 6
        full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + input_hyp + prompts[4] + structured_extra_knowledge[1] + prompts[5]
        # structured_gene: [hyp, reasoning process]
        sturctured_hyp_gene = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Hypothesis:'], if_only_return_one_structured_gene_component=True, restructure_output_model_name=self.args.model_name)
        sturctured_hyp_gene = exchange_order_in_list(sturctured_hyp_gene)
        ## provide feedback to hypothesis
        prompts = instruction_prompts("provide_feedback_to_hypothesis_four_aspects_with_extra_knowledge")
        assert len(prompts) == 6
        full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + structured_extra_knowledge[1] + prompts[4] + sturctured_hyp_gene[0] + prompts[5]
        feedback = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=False, restructure_output_model_name=self.args.model_name)
        ## hypothesis refinement
        prompts = instruction_prompts("hypothesis_refinement_with_feedback_with_extra_knowledge")
        assert len(prompts) == 7
        full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + structured_extra_knowledge[1] + prompts[4] + sturctured_hyp_gene[0] + prompts[5] + feedback + prompts[6]
        # structured_gene: [hyp, reasoning process]
        sturctured_hyp_gene_refined = llm_generation_while_loop(full_prompt, self.args.model_name, self.client, if_structured_generation=True, template=['Reasoning Process:', 'Refined Hypothesis:'], if_only_return_one_structured_gene_component=True, restructure_output_model_name=self.args.model_name)
        sturctured_hyp_gene_refined = exchange_order_in_list(sturctured_hyp_gene_refined)
        # hypothesis_collection: [extra_knowledge_0, output_hyp_0, reasoning_process_0, feedback_0, refined_hyp_0]
        hypothesis_collection = [structured_extra_knowledge[1], sturctured_hyp_gene[0], sturctured_hyp_gene[1], feedback, sturctured_hyp_gene_refined[0], sturctured_hyp_gene_refined[1]]
        return structured_extra_knowledge[0], hypothesis_collection
        
    


    ## Function
    # A single search step that combines bkg and insp to generate one single hypothesis, and nothing more (no refinement)
    ## Input
    # bkg_q/bkg_survey: text
    # insp_core_node: [title, reason, abstract]
    # same_mutation_prev_hyp: text
    # hyp_feedback: text
    # other_mutations (recombination_type=0/1): [hyp0, hyp1, ...], here 0, 1 indicates the id of mutation, for 
    # other_mutations (recombination_type=2): [insp_title, insp_abstract, hyp], for 
    # recombination_type: 0/1/2; 
    #   2: recombinational mutation between the final hypothesis developed from (the same background and) different inspirations;
    #   1: recombinational mutation between the last refined hypothesis over each mutation line. Each line is developed from the same background and the same inspiration; 
    #   0: develop a new mutation line
    #   -1: no recombinational mutation at all, even no combination between background and inspiration, since we don't retrieve inspirations for this baseline (baseline_type 2)
    # diffence between same_mutation_prev_hyp and other_mutations: same_mutation_prev_hyp follow the same type of mutation, and other_mutations are different types of mutations
    # this_mutation: text of a hypothesis (already developed based on cur_insp_core_node); only used when recombination_type=2
    ## Output
    # structured_gene: [hyp, reasoning process]
    def one_inference_for_one_hyp_gene(self, backgroud_question, backgroud_survey, cur_insp_core_node, same_mutation_prev_hyp=None, hyp_feedback=None, other_mutations=None, recombination_type=0, this_mutation=None):
        # check input
        assert recombination_type in [0, 1, 2]
        # set recombination_type to -1 when the baseline_type is 2
        if self.args.baseline_type == 2:
            # baseline 2 should be based on MOOSE, so here the received recombination_type must be 0
            assert recombination_type == 0
            # -1: no recombinational mutation at all, even no combination between background and inspiration, since we don't retrieve inspirations for this baseline (baseline_type 2)
            recombination_type = -1
        if hyp_feedback != None:
            assert same_mutation_prev_hyp != None
        # this_mutation is used iff recombination_type=2
        assert this_mutation == None if recombination_type != 2 else this_mutation != None
            
        # core insp prompt
        cur_insp_core_node_prompt = "title: {}; abstract: {}; one of the potential reasons on why this inspiration could be helpful: {}.".format(cur_insp_core_node[0], cur_insp_core_node[2], cur_insp_core_node[1])
        
        
        ## instructions
        if recombination_type == 2:
            assert other_mutations != None
            # other_mutations: [insp_title, insp_abstract, hyp]
            assert len(other_mutations) == 3
            # other_mutations_prompt
            other_mutations_prompt = "The selected complementary inspiration has title: {}, and abstract: {}. This complementary inspiration can lead to the hypothesis: {}. This hypothesis could be useful to understand how this complementary inspiration can be helpful.".format(other_mutations[0], other_mutations[1], other_mutations[2])
            if same_mutation_prev_hyp == None and hyp_feedback == None:
                prompts = instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration")
                assert len(prompts) == 6
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + this_mutation + prompts[4] + other_mutations_prompt + prompts[5]
                template = ['Reasoning Process:', 'Hypothesis:']
            # during refinement of recombination of different insp islands, we need to keep seeing the different hyps from different insp islands
            elif same_mutation_prev_hyp != None and hyp_feedback != None:
                prompts = instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration_with_feedback")
                assert len(prompts) == 8
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + this_mutation + prompts[4] + other_mutations_prompt + prompts[5] + same_mutation_prev_hyp + prompts[6] + hyp_feedback + prompts[7]
                template = ['Reasoning Process:', 'Refined Hypothesis:']
            else:
                raise ValueError("should not have this case. same_mutation_prev_hyp: {}; hyp_feedback: {}".format(same_mutation_prev_hyp, hyp_feedback))
        elif recombination_type == 1:
            assert other_mutations != None
            # other_mutations_prompt
            other_mutations_prompt = ""
            for cur_other_mutation_id, cur_other_mutation in enumerate(other_mutations):
                cur_other_mutation_prompt = "Next is previous hypothesis {}: {}.\n".format(cur_other_mutation_id, cur_other_mutation)
                other_mutations_prompt += cur_other_mutation_prompt
            # instructions
            if same_mutation_prev_hyp == None and hyp_feedback == None:
                prompts = instruction_prompts("final_recombinational_mutation_hyp_gene_same_bkg_insp")
                assert len(prompts) == 5
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + other_mutations_prompt + prompts[4]
                template = ['Reasoning Process:', 'Hypothesis:']
            elif same_mutation_prev_hyp != None and hyp_feedback != None:
                prompts = instruction_prompts("final_recombinational_mutation_hyp_gene_same_bkg_insp_with_feedback")
                assert len(prompts) == 7
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + other_mutations_prompt + prompts[4] + same_mutation_prev_hyp + prompts[5] + hyp_feedback + prompts[6]
                template = ['Reasoning Process:', 'Refined Hypothesis:']
            else:
                raise ValueError("should not have this case. same_mutation_prev_hyp: {}; hyp_feedback: {}".format(same_mutation_prev_hyp, hyp_feedback))
        elif recombination_type == 0:
            # other_mutations_prompt
            if other_mutations != None:
                other_mutations_prompt = ""
                for cur_other_mutation_id, cur_other_mutation in enumerate(other_mutations):
                    cur_other_mutation_prompt = "Next is previous hypothesis {}: {}.\n".format(cur_other_mutation_id, cur_other_mutation)
                    other_mutations_prompt += cur_other_mutation_prompt
            # instructions
            if other_mutations == None and hyp_feedback == None:
                prompts = instruction_prompts("coarse_hypothesis_generation_only_core_inspiration")
                assert len(prompts) == 4
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3]
                template=['Reasoning Process:', 'Hypothesis:']
            elif other_mutations == None and hyp_feedback != None:
                prompts = instruction_prompts("hypothesis_generation_with_feedback_only_core_inspiration")
                assert len(prompts) == 6
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + same_mutation_prev_hyp + prompts[4] + hyp_feedback + prompts[5]
                template=['Reasoning Process:', 'Refined Hypothesis:']
            # to develop the second or more mutation line that should be different with the hypotheses from previous mutation lines
            elif other_mutations != None and hyp_feedback == None:
                prompts = instruction_prompts("hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration")
                assert len(prompts) == 5
                # full_prompt
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + cur_insp_core_node_prompt + prompts[3] + other_mutations_prompt + prompts[4]
                template = ['Reasoning Process:', 'Hypothesis:']
            elif other_mutations != None and hyp_feedback != None:
                raise ValueError("should not have both other_mutations and hyp_feedback")
            else:
                raise ValueError("should not have this case")   
        elif recombination_type == -1:
            assert other_mutations == None
            if hyp_feedback == None:
                prompts = instruction_prompts("coarse_hypothesis_generation_without_inspiration")
                assert len(prompts) == 3
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] 
                template = ['Reasoning Process:', 'Hypothesis:']
            elif hyp_feedback != None:
                prompts = instruction_prompts("hypothesis_generation_with_feedback_without_inspiration")
                assert len(prompts) == 5
                full_prompt = prompts[0] + backgroud_question + prompts[1] + backgroud_survey + prompts[2] + same_mutation_prev_hyp + prompts[3] + hyp_feedback + prompts[4]
                template = ['Reasoning Process:', 'Refined Hypothesis:']
            else:
                raise ValueError("should not have this case")
        else:
            raise ValueError("recombination_type: {} is not supported".format(recombination_type))
             
        ## generation
        while True:
            try:
                cur_gene = llm_generation(full_prompt, self.args.model_name, self.client)
                cur_structured_gene = get_structured_generation_from_raw_generation(cur_gene, template=template)
                cur_structured_gene = exchange_order_in_list(cur_structured_gene)
                break
            except AssertionError as e:
                # if the format
                print("AssertionError: {}, try again..".format(e))
        
        # cur_structured_gene: [[hyp, reasoning process]] --> [hyp, reasoning process]
        assert len(cur_structured_gene) == 1 and len(cur_structured_gene[0]) == 2
        cur_structured_gene = cur_structured_gene[0]
        return cur_structured_gene



    ## Function
    # provide textual feedback (including suggestions) to a hypothesis
    ## Input
    # cur_hypothesis_and_reasoning_process: [hyp, reasoning process]
    ## Output
    # feedback: text
    def hypothesis_refinement(self, cur_hypothesis_and_reasoning_process, if_with_external_knowledge_feedback=False):
        # cur_hypothesis_and_reasoning_process_prompt
        cur_hypothesis_and_reasoning_process_prompt = "hypothesis: {}; reasoning process: {}.".format(cur_hypothesis_and_reasoning_process[0], cur_hypothesis_and_reasoning_process[1])
        # instructions
        if self.args.baseline_type == 1:
            # only novelty checker, no reality checker nor clarity checker (Scimon)
            prompts = instruction_prompts("novelty_checking")
        elif self.args.baseline_type == 3:
            # no significance checker, but only novelty, reality, and clarity checker
            prompts = instruction_prompts("three_aspects_checking_no_significance")
        else:
            # normal setting
            if if_with_external_knowledge_feedback:
                prompts = instruction_prompts("four_aspects_checking_and_extra_knowledge")
            else:
                prompts = instruction_prompts("four_aspects_checking")

        assert len(prompts) == 2
        full_prompt = prompts[0] + cur_hypothesis_and_reasoning_process_prompt + prompts[1]
        # generation
        while True:
            try:
                feedback = llm_generation(full_prompt, self.args.model_name, self.client)
                break
            except AssertionError as e:
                # if the format
                print("AssertionError: {}, try again..".format(e))
        return feedback
        

    ## Function
    # provide numerical evaluation for a hypothesis (mainly for evaluation, but not further refine)
    ## Input
    # cur_hypothesis_and_reasoning_process: [hyp, reasoning process] or [hyp, reasoning process, feedback]
    ## Output
    # score_collection: ['score0', 'score1', 'score2', 'score3']
    # score_reason_collection: ['reason0', 'reason1', 'reason2', 'reason3']
    def hypothesis_evaluation(self, cur_hypothesis_and_reasoning_process):
        # cur_hypothesis_prompt: for evaluation, we only need the hypothesis itself, but not reasoning process
        cur_hypothesis_prompt = "hypothesis: {}.".format(cur_hypothesis_and_reasoning_process[0])
        # instructions
        prompts = instruction_prompts("four_aspects_self_numerical_evaluation")
        assert len(prompts) == 2
        full_prompt = prompts[0] + cur_hypothesis_prompt + prompts[1]
        # generation
        while True:
            try:
                score_text = llm_generation(full_prompt, self.args.model_name, self.client)
                score_collection, score_reason_collection, if_successful = pick_score(score_text)
                assert if_successful == True
                break
            except AssertionError as e:
                print(f"Warning: pick_score failed, score_text: {score_text}")
                # if the format
                print("AssertionError: {}, try again..".format(e))
            except Exception as e:
                print(f"Warning: pick_score failed, score_text: {score_text}")
                print("Exception: {}, try again..".format(e))
        return score_collection, score_reason_collection
    

    def save_file(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print("Saved data to {}".format(file_path))






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hypothesis generation')
    parser.add_argument("--model_name", type=str, default="chatgpt", help="model name: gpt4/chatgpt/chatgpt16k/claude35S/gemini15P/llama318b/llama3170b/llama31405b")
    parser.add_argument("--api_type", type=int, default=1, help="0: openai's API toolkit; 1: azure's API toolkit")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="https://api.claudeshop.top/v1", help="base url for the API")
    parser.add_argument("--chem_annotation_path", type=str, default="./chem_research_2024.xlsx", help="store annotated background research questions and their annotated groundtruth inspiration paper titles")
    parser.add_argument("--if_use_background_survey", type=int, default=1, help="whether use background survey. 0: not use (replace the survey as 'Survey not provided. Please overlook the survey.'); 1: use")
    parser.add_argument("--if_use_strict_survey_question", type=int, default=1, help="whether to use the strict version of background survey and background question. strict version means the background should not have any close information to inspirations and the hypothesis, even if the close information is a commonly used method in that particular background question domain.")
    parser.add_argument("--custom_research_background_path", type=str, default="", help="the path to the research background file. The format is [research question, background survey], and saved in a json file. ")
    parser.add_argument("--custom_inspiration_corpus_path", type=str, default="", help="store title and abstract of the inspiration corpus; Should be a json file in a format of [[title, abstract], ...]; It will be automatically assigned with a default value if it is not assigned by users. The default value is './Data/Inspiration_Corpus_{}.json'.format(args.corpus_size). (The default value is the groundtruth inspiration papers for the Tomato-Chem Benchmark and random high-quality papers)")
    parser.add_argument("--inspiration_dir", type=str, default="./Checkpoints/coarse_inspiration_search_gpt4.json;", help="store coarse-grained inspiration screening results")
    parser.add_argument("--output_dir", type=str, default="./Checkpoints/hypothesis_generation_results.json")
    parser.add_argument("--if_save", type=int, default=0, help="whether save grouping results")
    parser.add_argument("--if_load_from_saved", type=int, default=0, help="whether load data that is previous to inter-EA recombination; when used, the framework will load data from output_dir, instead of generating from scratch; mainly used for debugging and improving inter-EA recombination") 
    parser.add_argument("--background_question_id", type=int, default=0, help="the background question id in background literatures. Since running for one background costs enough api callings, we only run for one background question at a time.")
    # parser.add_argument("--inspiration_id", type=int, default=0, help="the inspiration id in the background question. -1: iterate over all inspirations; otherwise: only generate hypothesis for one inspiration")
    parser.add_argument(
        "--inspiration_ids",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[0],  # default if nothing is provided
        help="the inspiration id in the background question, used for determine which inspirations will be used to develop hypotheses. -1: iterate over all inspirations; otherwise: only generate hypothesis for specified inspiration ids"
    )
    parser.add_argument(
        "--recom_inspiration_ids",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[],  # default if nothing is provided
        help="the inspiration ids in the background question, used for determine which inspiration and its hypothesis will perform inter-EA recombination. -1: iterate over all inspirations; otherwise: only generate hypothesis for specified inspiration ids"
    )
    parser.add_argument(
        "--self_explore_inspiration_ids",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[],  # default if nothing is provided
        help="the inspiration ids in the background question, used for determine which inspiration and its hypothesis will perform self-exploration. -1: iterate over all inspirations; otherwise: only generate hypothesis for specified inspiration ids"
    )
    parser.add_argument("--if_only_core_inspiration", type=int, default=1, help="whether only use core inspiration to generate hypothesis (no inspiration grouping).")
    parser.add_argument("--if_mutate_inside_same_bkg_insp", type=int, default=0, help="whether we perform mutation inside the same bkg and insp pair; do not control self-explore, since self-explore will perform self-mutate regardless of if_mutate_inside_same_bkg_insp")
    parser.add_argument("--if_mutate_between_diff_insp", type=int, default=0, help="whether we perform mutation between different insp nodes (different insp nodes have the same bkg).")
    parser.add_argument("--if_self_explore", type=int, default=0, help="whether we perform self-explore extra knowledge for the given bkg with background_question_id and inspirations with self_explore_inspiration_ids.")
    parser.add_argument("--num_mutations", type=int, default=3, help="how many mutations we generate with the same pair of bkg and insp; can control the num_mutations in self-explore")
    parser.add_argument("--num_itr_self_refine", type=int, default=2, help="how many refinements we do for each hypothesis generation step")
    parser.add_argument("--num_self_explore_steps_each_line", type=int, default=3, help="how many times we (self-explore extra knowledge and generate hypothesis and refine them), for each mutation line from one original hypothesis node; only control self-explore step")
    parser.add_argument("--num_screening_window_size", type=int, default=12, help="how many (insp, hyp) pairs we use in a single inference of llm to screen inspiration candidates during second/third/... round of inspiration screening")
    parser.add_argument("--num_screening_keep_size", type=int, default=3, help="how many (insp, hyp) pairs we keep out of --num_screening_window_size")
    parser.add_argument("--recom_num_beam_size", type=int, default=10, help="beam size of inspirations that will perform inter-EA recombination; only used when recom_inspiration_ids is empty")
    parser.add_argument("--self_explore_num_beam_size", type=int, default=10, help="beam size of inspirations that will perform self-exploration; only used when self_explore_inspiration_ids is empty")
    parser.add_argument("--idx_round_of_first_step_insp_screening", type=int, default=0, help="which round of screened inspirations we use.")
    parser.add_argument("--max_inspiration_search_steps", type=int, default=2, help="how many steps we screen inspirations for a complex hypothesis that require multiple keypoints (one inspiration brings in one keypoint)")
    parser.add_argument("--if_use_gdth_insp", type=int, default=0, help="whether directly load groundtruth inspirations (instead of screened inspirations) for hypothesis generation")
    parser.add_argument("--if_consider_external_knowledge_feedback_during_second_refinement", type=int, default=0, help="during the second hypothsis refinement, whether the feedback to hypothesis will consider to add external knowledge to make the hypothesis more complete")
    parser.add_argument("--corpus_size", type=int, default=300, help="the number of total inspiration (paper) corpus (both groundtruth insp papers and non-groundtruth insp papers)")
    parser.add_argument("--baseline_type", type=int, default=0, help="0: not using baseline; 1: MOOSE w/o novelty and clarity checker (Scimon); 2. MOOSE w/o novelty retrieval (<Large Language Models are Zero Shot Hypothesis Proposers>); 3: MOOSE-Chem w/o significance checker")
    args = parser.parse_args()

    assert args.api_type in [0, 1]
    assert args.if_use_background_survey in [0, 1]
    assert args.if_use_strict_survey_question in [0, 1]
    assert args.if_save in [1]
    assert args.if_load_from_saved in [0, 1]
    # we don't use additional inspiration anymore, since they might not be necessary, and is not very helpful, from the inspiration grouping results (always choose the two or three of inspirations)
    assert args.if_only_core_inspiration in [1]
    assert args.if_mutate_inside_same_bkg_insp in [0, 1]
    assert args.if_mutate_between_diff_insp in [0, 1]
    assert args.if_self_explore in [0, 1]
    # currently cannot adjust corresponding prompts by args.num_screening_keep_size (default prompt is three, else need to change the prompt)
    assert args.num_screening_keep_size in [3]
    assert args.if_use_gdth_insp in [0, 1]
    assert args.if_consider_external_knowledge_feedback_during_second_refinement in [0, 1]
    assert args.baseline_type in [0, 1, 2, 3]
    if args.baseline_type not in [0, 3]:
        print("Warning: Running baseline {}..".format(args.baseline_type))
        # the baseline is based on MOOSE, not MOOSE-Chem, so we set up the parameters for MOOSE
        assert args.if_mutate_inside_same_bkg_insp == 0 and args.if_mutate_between_diff_insp == 0 and args.if_self_explore == 0
    # args.output_dir = os.path.abspath(args.output_dir)

    ## Setup logger
    logger = setup_logger(args.output_dir)
    # Redirect print to logger
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args))
        logger.info(message)
    # global print
    # print = custom_print
    builtins.print = custom_print
    print("args: ", args)

    ## initialize research question and background survey to text to use them for inference (by default they are set to those in the Tomato-Chem benchmark)
    if args.custom_research_background_path == "":
        custom_rq, custom_bs = None, None
        print("Using the research background in the Tomato-Chem benchmark.")
    else:
        assert os.path.exists(args.custom_research_background_path), "The research background file does not exist: {}".format(args.custom_research_background_path)
        with open(args.custom_research_background_path, 'r') as f:
            research_background = json.load(f)
        # research_background: [research question, background survey]
        assert len(research_background) == 2
        assert isinstance(research_background[0], str) and isinstance(research_background[1], str)
        custom_rq = research_background[0]
        custom_bs = research_background[1]
        print("Using custom research background. \nResearch question: \n{}; \n\nBackground survey: \n{}".format(custom_rq, custom_bs))

    ## change inspiration corpus path to the default corpus if it is not assigned by users
    if args.custom_inspiration_corpus_path == "":
        args.custom_inspiration_corpus_path = './Data/Inspiration_Corpus_{}.json'.format(args.corpus_size)
        print("Using the default inspiration corpus: {}".format(args.custom_inspiration_corpus_path))
    else:
        assert os.path.exists(args.custom_inspiration_corpus_path), "The inspiration corpus file does not exist: {}".format(args.custom_inspiration_corpus_path)
        print("Using custom inspiration corpus: {}".format(args.custom_inspiration_corpus_path))


    start_time = time.time()

    # load from existing collection
    if args.if_load_from_saved == 1:
        with open(args.output_dir, 'r') as f:
            final_data_collection = json.load(f)
    else:
        final_data_collection = None
   
    # skip if the output_dir already exists
    # Q: overlook args.if_load_from_saved for recent experiments
    if os.path.exists(args.output_dir):
        print("Warning: {} already exists.".format(args.output_dir))
    else:
        # initialize an object
        hyp_gene_ea = HypothesisGenerationEA(args, custom_rq=custom_rq, custom_bs=custom_bs)
        # hypothesis generation for one background question
        final_data_collection = hyp_gene_ea.hypothesis_generation_for_one_background_question(background_question_id=args.background_question_id, inspiration_ids=args.inspiration_ids, final_data_collection=final_data_collection)

    duration = time.time() - start_time
    
    print("Finished within {} seconds!".format(duration))