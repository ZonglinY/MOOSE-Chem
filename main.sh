#!/bin/bash
#SBATCH -J test    
#SBATCH -o logs/test.out 
#SBATCH -e logs/test.out   
#SBATCH -p AI4Chem        
#SBATCH -N 1               
#SBATCH -n 1               
#SBATCH --gres=gpu:0



api_key="sk-"
base_url=""

model_name_insp_retrieval=""
model_name_gene=""
model_name_eval=""

checkpoint_root_dir="./Checkpoints/"
display_txt_file_path="./hypothesis.txt"
output_dir_postfix=""


# custom_research_background_path: set to "" if you want to use the default research background in TOMATO-Bench
custom_research_background_path=""
# custom_raw_inspiration_data_dir: raw custom inspiration data to process to inspiration corpus
custom_raw_inspiration_data_dir=""
# custom_inspiration_corpus_path: set to "" if you want to use the default inspiration corpus in TOMATO-Bench
custom_inspiration_corpus_path=""



# ## Custom Research Background Dumping 
# #   if use custom research background, please modify the 'research_question' and 'background_survey' in research_background_to_json() function in custom_research_background_dumping_and_output_displaying.py
# python -u ./Preprocessing/custom_research_background_dumping_and_output_displaying.py --io_type 0 \
#         --custom_research_background_path ${custom_research_background_path} 



# ## Custom Inspiration Corpus Dumping
# python -u ./Preprocessing/construct_custom_inspiration_corpus.py \
#         --raw_data_dir ${custom_raw_inspiration_data_dir} \
#         --custom_inspiration_corpus_path ${custom_inspiration_corpus_path}



## Inspiration Retrieval
# --custom_research_background_path: Path to custom research question and background survey.
#    Leave empty ("") to use the default from TOMATO-Bench.
# --custom_inspiration_corpus_path: Path to custom inspiration corpus.
#    Leave empty ("") to use the default corpus controlled by --corpus_size.
python -u ./Method/inspiration_screening.py --model_name ${model_name_insp_retrieval} \
        --api_type 1 --api_key ${api_key} --base_url ${base_url} \
        --chem_annotation_path ./Data/chem_research_2024.xlsx \
        --output_dir ${checkpoint_root_dir}/coarse_inspiration_search_${model_name_insp_retrieval}_${output_dir_postfix}.json \
        --corpus_size 150 --if_use_background_survey 1 --if_use_strict_survey_question 1 \
        --num_screening_window_size 15 --num_screening_keep_size 3 --num_round_of_screening 4 \
        --if_save 1 --background_question_id 0 --if_select_based_on_similarity 0  \
        --custom_research_background_path ${custom_research_background_path} \
        --custom_inspiration_corpus_path ${custom_inspiration_corpus_path}



## Hypothesis Composition
# --custom_research_background_path: Path to custom research question and background survey.
#    Leave empty ("") to use the default from TOMATO-Bench.
# --custom_inspiration_corpus_path: Path to custom inspiration corpus.
#    Leave empty ("") to use the default corpus controlled by --corpus_size.
python -u ./Method/hypothesis_generation.py --model_name ${model_name_gene} \
        --api_type 1 --api_key ${api_key} --base_url ${base_url} \
        --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 150 --if_use_strict_survey_question 1 --if_use_background_survey 1 \
        --inspiration_dir ${checkpoint_root_dir}/coarse_inspiration_search_${model_name_insp_retrieval}_${output_dir_postfix}.json \
        --output_dir ${checkpoint_root_dir}/hypothesis_generation_${model_name_gene}_${output_dir_postfix}.json \
        --if_save 1 --if_load_from_saved 0 \
        --if_use_gdth_insp 0 --idx_round_of_first_step_insp_screening 2 \
        --num_mutations 3 --num_itr_self_refine 3  --num_self_explore_steps_each_line 3 --num_screening_window_size 12 --num_screening_keep_size 3 \
        --if_mutate_inside_same_bkg_insp 1 --if_mutate_between_diff_insp 1 --if_self_explore 0 --if_consider_external_knowledge_feedback_during_second_refinement 0 \
        --inspiration_ids -1  --recom_inspiration_ids  --recom_num_beam_size 5  --self_explore_inspiration_ids   --self_explore_num_beam_size 5 \
        --max_inspiration_search_steps 3 --background_question_id 0  \
        --custom_research_background_path ${custom_research_background_path} \
        --custom_inspiration_corpus_path ${custom_inspiration_corpus_path}



## Hypothesis Ranking
# --custom_inspiration_corpus_path: Path to custom inspiration corpus.
#    Leave empty ("") to use the default corpus controlled by --corpus_size.
python -u ./Method/evaluate.py --model_name ${model_name_eval} \
        --api_type 1 --api_key ${api_key} --base_url ${base_url} \
        --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 150 \
        --hypothesis_dir ${checkpoint_root_dir}/hypothesis_generation_${model_name_gene}_${output_dir_postfix}.json \
        --output_dir ${checkpoint_root_dir}/evaluation_${model_name_eval}_${output_dir_postfix}.json \
        --if_save 1 --if_load_from_saved 0 \
        --if_with_gdth_hyp_annotation 0 \
        --custom_inspiration_corpus_path ${custom_inspiration_corpus_path} 



# ## Hypothesis Display
# python -u ./Preprocessing/custom_research_background_dumping_and_output_displaying.py --io_type 1 \
#         --evaluate_output_dir ${checkpoint_root_dir}/evaluation_${model_name_eval}_${output_dir_postfix}.json \
#         --display_dir ${display_txt_file_path}




## Analysis: Ranking Groundtruth Hypothesis Between Generated Hypothesis
# python -u ./Analysis/groundtruth_hyp_ranking.py --model_name ${model_name} \
#         --api_type 0 --api_key ${api_key} --base_url ${base_url} \
#         --evaluate_result_dir ${checkpoint_root_dir}/evaluation_${model_name}_corpus_150_survey_1_gdthInsp_1_intraEA_1_interEA_1_bkgid_ \
#         --if_save 1 --output_dir ${checkpoint_root_dir}/groundtruth_hypothesis_automatic_scores_four_aspects_${model_name}.json



