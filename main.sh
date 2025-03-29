#!/bin/bash
#SBATCH -J test    
#SBATCH -o logs/test.out 
#SBATCH -e logs/test.out   
#SBATCH -p AI4Chem        
#SBATCH -N 1               
#SBATCH -n 1               
#SBATCH --gres=gpu:0



api_key="sk-"
base_url="https://api.claudeshop.top/v1"


# # coarse_inspiration_search_gpt4_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_0
# --title_abstract_all_insp_literature_path (inspiration corpus dir; if not provided, use the default one involving args.corpus_size)
# python -u ./Method/inspiration_screening.py --model_name gpt4 \
#         --api_type 0 --api_key ${api_key} --base_url ${base_url} \
#         --chem_annotation_path ./Data/chem_research_2024.xlsx \
#         --output_dir ./Checkpoints/coarse_inspiration_search_gpt4_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_0.json \
#         --corpus_size 300 --if_use_background_survey 1 --if_use_strict_survey_question 1 \
#         --num_screening_window_size 15 --num_screening_keep_size 3 --num_round_of_screening 4 \
#         --if_save 1 --background_question_id 0 --if_select_based_on_similarity 0  \
        
        


# # # hypothesis_generation_gpt4_corpus_300_survey_1_gdthInsp_0_intraEA_1_interEA_1_bkgid_0
# --title_abstract_all_insp_literature_path (inspiration corpus dir; if not provided, use the default one involving args.corpus_size)
# python -u ./Method/hypothesis_generation.py --model_name gpt4 \
#         --api_type 0 --api_key ${api_key} --base_url ${base_url} \
#         --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 --if_use_strict_survey_question 1 --if_use_background_survey 1 \
#         --inspiration_dir ./Checkpoints/coarse_inspiration_search_gpt4_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_0.json \
#         --output_dir ./Checkpoints/hypothesis_generation_gpt4_corpus_300_survey_1_gdthInsp_0_intraEA_1_interEA_1_bkgid_0.json \
#         --if_save 1 --if_load_from_saved 0 \
#         --if_use_gdth_insp 0 --idx_round_of_first_step_insp_screening 1 \
#         --num_mutations 3 --num_itr_self_refine 3  --num_self_explore_steps_each_line 3 --num_screening_window_size 12 --num_screening_keep_size 3 \
#         --if_mutate_inside_same_bkg_insp 1 --if_mutate_between_diff_insp 1 --if_self_explore 0 --if_consider_external_knowledge_feedback_during_second_refinement 0 \
#         --inspiration_ids -1  --recom_inspiration_ids  --recom_num_beam_size 15  --self_explore_inspiration_ids   --self_explore_num_beam_size 15 \
#         --max_inspiration_search_steps 3 --background_question_id 0  \
        


# # evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_intraEA_1_interEA_1_bkgid_0
# --title_abstract_all_insp_literature_path (inspiration corpus dir; if not provided, use the default one involving args.corpus_size)
# python -u ./Method/evaluate.py --model_name gpt4 \
#         --api_type 0 --api_key ${api_key} --base_url ${base_url} \
#         --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 \
#         --hypothesis_dir ./Checkpoints/hypothesis_generation_gpt4_corpus_300_survey_1_gdthInsp_0_intraEA_1_interEA_1_bkgid_0.json \
#         --output_dir ./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_0_intraEA_1_interEA_1_bkgid_0.json \
#         --if_save 1 --if_load_from_saved 0 \
#         --if_with_gdth_hyp_annotation 1 \



# evaluation_gpt4_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_
# model_name=gpt4
# python -u ./Analysis/groundtruth_hyp_ranking.py --model_name ${model_name} \
#         --api_type 0 --api_key ${api_key} --base_url ${base_url} \
#         --evaluate_result_dir ./Checkpoints/evaluation_gpt4_corpus_300_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_ \
#         --if_save 1 --output_dir ./Checkpoints/groundtruth_hypothesis_automatic_scores_four_aspects_${model_name}.json



# HF_HUB_OFFLINE=1
# python try.py