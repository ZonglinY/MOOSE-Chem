#!/bin/bash
#SBATCH -J exp8
#SBATCH -o logs/exp8
#SBATCH -e logs/exp8
#SBATCH -p AI4Chem        
#SBATCH -N 1                
#SBATCH -n 1                
#SBATCH --gres=gpu:0


api_key="sk-"


## Experiment 5: Main Table for Assumption 2 (Main Table 1 and Main Table 2) (when if_use_background_survey == 1)
## Experiment 6: Alation Table on the role of background survey (when if_use_background_survey == 0)

# for if_use_background_survey in 1 0
# do
#         echo "\n\nif_use_background_survey: $if_use_background_survey"
#         for bkg_q_id in {0..50}
#         do
#                 echo "\n\nEntering loop for bkg_q_id: $bkg_q_id"
#                 python -u hypothesis_generation.py --model_name gpt4 \
#                         --api_type 0 --api_key ${api_key} \
#                         --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 --if_use_strict_survey_question 1 --if_use_background_survey ${if_use_background_survey} \
#                         --inspiration_dir ./Checkpoints/coarse_inspiration_search_gpt4_corpusSize_300_survey_${if_use_background_survey}_strict_1_numScreen_15_round_4_similarity_0_bkgid_${bkg_q_id}.json \
#                         --output_dir ./Checkpoints/hypothesis_generation_gpt4_corpus_300_survey_${if_use_background_survey}_gdthInsp_1_intraEA_1_interEA_1_bkgid_${bkg_q_id}.json \
#                         --if_save 1 --if_load_from_saved 0 \
#                         --if_use_gdth_insp 1 --idx_round_of_first_step_insp_screening 0 \
#                         --num_mutations 3 --num_itr_self_refine 3  --num_self_explore_steps_each_line 3 --num_screening_window_size 12 --num_screening_keep_size 3 \
#                         --if_mutate_inside_same_bkg_insp 1 --if_mutate_between_diff_insp 1 --if_self_explore 0 --if_consider_external_knowledge_feedback_during_second_refinement 0 \
#                         --inspiration_ids -1  --recom_inspiration_ids -1 --recom_num_beam_size 10  --self_explore_inspiration_ids  --self_explore_num_beam_size 10 \
#                         --max_inspiration_search_steps 3 --background_question_id ${bkg_q_id} 
                        
#                 echo "\n\nRunning evaluate.py for bkg_q_id: $bkg_q_id"
#                 python -u evaluate.py --model_name gpt4 \
#                         --api_type 0 --api_key ${api_key} \
#                         --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 \
#                         --hypothesis_dir ./Checkpoints/hypothesis_generation_gpt4_corpus_300_survey_${if_use_background_survey}_gdthInsp_1_intraEA_1_interEA_1_bkgid_${bkg_q_id}.json \
#                         --output_dir ./Checkpoints/evaluation_gpt4_corpus_300_survey_${if_use_background_survey}_gdthInsp_1_intraEA_1_interEA_1_bkgid_${bkg_q_id}.json \
#                         --if_save 1 --if_load_from_saved 0 
#         done
# done
# echo "Experiment 5 and Experiment 6 finished successfully"





## Experiment 7/8/9/10/11

experiment_id=11

## Experiment 7: MOOSE
if [[ ${experiment_id} -eq 7 ]]; then
        echo "Experiment 7: MOOSE"
        round_of_insp_screening=1
        recom_num_beam_size=15
        # intra-EA
        if_mutate_inside_same_bkg_insp=0
        # inter-EA
        if_mutate_between_diff_insp=0
        # not baseline
        baseline_type=0
## Experiment 8: MOOSE-Chem
elif [[ ${experiment_id} -eq 8 ]]; then
        echo "Experiment 8: MOOSE-Chem"
        # use round 1 to save inference time, can be an ablation to also test round 0
        round_of_insp_screening=1
        # maybe 15 beam size can be a good tradeoff between performance and inference time
        recom_num_beam_size=15
        # intra-EA
        if_mutate_inside_same_bkg_insp=1
        # inter-EA
        if_mutate_between_diff_insp=1
        # not baseline
        baseline_type=0
## Experiment 9: MOOSE-Chem w/o intraEA
elif [[ ${experiment_id} -eq 9 ]]; then
        echo "Experiment 9: MOOSE-Chem w/o intraEA"
        round_of_insp_screening=1
        recom_num_beam_size=15
        # intra-EA
        if_mutate_inside_same_bkg_insp=0
        # inter-EA
        if_mutate_between_diff_insp=1
        # not baseline
        baseline_type=0
## Experiment 10: (baseline 1) MOOSE w/o valid and clarity feedback 
elif [[ ${experiment_id} -eq 10 ]]; then
        echo "Experiment 10: MOOSE w/o valid and clarity feedback (baseline 1)"
        round_of_insp_screening=1
        recom_num_beam_size=15
        # intra-EA
        if_mutate_inside_same_bkg_insp=0
        # inter-EA
        if_mutate_between_diff_insp=0
        # baseline 1
        baseline_type=1
## Experiment 11: (baseline 2) MOOSE w/o inspiration retrieval 
elif [[ ${experiment_id} -eq 11 ]]; then
        echo "Experiment 11: MOOSE w/o inspiration retrieval (baseline 2)"
        round_of_insp_screening=1
        recom_num_beam_size=15
        # intra-EA
        if_mutate_inside_same_bkg_insp=0
        # inter-EA
        if_mutate_between_diff_insp=0
        # baseline 2
        baseline_type=2
else
        echo "Invalid experiment_id: ${experiment_id}"
        exit 1
fi


echo "Experiment begins..."
for bkg_q_id in {0..50}
do
        echo "\n\nEntering loop for bkg_q_id: $bkg_q_id"
        python -u hypothesis_generation.py --model_name gpt4 \
                --api_type 0 --api_key ${api_key} \
                --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 --if_use_strict_survey_question 1 --if_use_background_survey 1 \
                --inspiration_dir ./Checkpoints/coarse_inspiration_search_gpt4_corpusSize_300_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_${bkg_q_id}.json \
                --output_dir ./Checkpoints/hypothesis_generation_gpt4_baseline_${baseline_type}_corpus_300_survey_1_gdthInsp_0_roundInsp_${round_of_insp_screening}_intraEA_${if_mutate_inside_same_bkg_insp}_interEA_${if_mutate_between_diff_insp}_beamsize_${recom_num_beam_size}_bkgid_${bkg_q_id}.json \
                --if_save 1 --if_load_from_saved 0 \
                --if_use_gdth_insp 0 --idx_round_of_first_step_insp_screening ${round_of_insp_screening} \
                --num_mutations 3 --num_itr_self_refine 3  --num_self_explore_steps_each_line 3 --num_screening_window_size 12 --num_screening_keep_size 3 \
                --if_mutate_inside_same_bkg_insp ${if_mutate_inside_same_bkg_insp} --if_mutate_between_diff_insp ${if_mutate_between_diff_insp} --if_self_explore 0 --if_consider_external_knowledge_feedback_during_second_refinement 0 \
                --inspiration_ids -1  --recom_inspiration_ids  --recom_num_beam_size ${recom_num_beam_size}  --self_explore_inspiration_ids  --self_explore_num_beam_size 10 \
                --max_inspiration_search_steps 3 --background_question_id ${bkg_q_id} \
                --baseline_type ${baseline_type}
                
        echo "\n\nRunning evaluate.py for bkg_q_id: $bkg_q_id"
        python -u evaluate.py --model_name gpt4 \
                --api_type 0 --api_key ${api_key} \
                --chem_annotation_path ./Data/chem_research_2024.xlsx --corpus_size 300 \
                --hypothesis_dir ./Checkpoints/hypothesis_generation_gpt4_baseline_${baseline_type}_corpus_300_survey_1_gdthInsp_0_roundInsp_${round_of_insp_screening}_intraEA_${if_mutate_inside_same_bkg_insp}_interEA_${if_mutate_between_diff_insp}_beamsize_${recom_num_beam_size}_bkgid_${bkg_q_id}.json \
                --output_dir ./Checkpoints/evaluation_gpt4_baseline_${baseline_type}_corpus_300_survey_1_gdthInsp_0_roundInsp_${round_of_insp_screening}_intraEA_${if_mutate_inside_same_bkg_insp}_interEA_${if_mutate_between_diff_insp}_beamsize_${recom_num_beam_size}_bkgid_${bkg_q_id}.json \
                --if_save 1 --if_load_from_saved 0 
done
echo "Experiment ${experiment_id} finished successfully"



