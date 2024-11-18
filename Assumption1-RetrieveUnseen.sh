#!/bin/bash
#SBATCH -J exp1_70b
#SBATCH -o logs/exp1_70b
#SBATCH -e logs/exp1_70b
#SBATCH -p AI4Chem        
#SBATCH -N 1                
#SBATCH -n 1              
#SBATCH --gres=gpu:0


api_key="sk-"
model_name_insp_retrieval="llama3170b"


## Define the base command, that is shared across all experiments
base_command="python -u inspiration_screening.py --model_name ${model_name_insp_retrieval} \
        --api_type 0 --api_key ${api_key} \
        --chem_annotation_path ./Data/chem_research_2024.xlsx \
        --output_dir ./Checkpoints/coarse_inspiration_search"



# Experiment 1: Main Table for Assumption 1

for corpus_size in 300
do
        echo corpus_size: $corpus_size
        for id in {0..50}
        do
                echo "Screening inspiration for id: $id"
                # Construct the full command for each iteration
                full_command="${base_command}_${model_name_insp_retrieval}_corpusSize_${corpus_size}_survey_1_strict_1_numScreen_15_round_4_similarity_0_bkgid_${id}.json \
                        --corpus_size ${corpus_size} --if_use_background_survey 1 --if_use_strict_survey_question 1 \
                        --num_screening_window_size 15 --num_screening_keep_size 3 --num_round_of_screening 4 \
                        --if_save 1 --background_question_id ${id} --if_select_based_on_similarity 0"

                # Execute the command
                eval "$full_command"
        done
        
done
echo "Experiment 1 finished successfully"



# ## Experiment 2: Ablation Table on num_screen (select 3 out of x)

# for num_screen in 10 20 40 60
# do
#         echo num_screen: $num_screen
#         for id in {0..50}
#         do
#                 echo "Screening inspiration for id: $id"
#                 # Construct the full command for each iteration
#                 full_command="${base_command}_${model_name_insp_retrieval}_corpusSize_300_survey_1_strict_1_numScreen_${num_screen}_round_4_similarity_0_bkgid_${id}.json \
#                         --corpus_size 300 --if_use_background_survey 1 --if_use_strict_survey_question 1 \
#                         --num_screening_window_size ${num_screen} --num_screening_keep_size 3 --num_round_of_screening 4 \
#                         --if_save 1 --background_question_id ${id} --if_select_based_on_similarity 0"

#                 # Execute the command
#                 eval "$full_command"
#         done
# done
# echo "Experiment 2 finished successfully"



# ## Experiment 3: Alation Table on the role of background survey

# if_survey=0

# for id in {0..50}
# do
#         echo "Screening inspiration for id: $id"
#         # Construct the full command for each iteration
#         full_command="${base_command}_${model_name_insp_retrieval}_corpusSize_300_survey_${if_survey}_strict_1_numScreen_15_round_4_similarity_0_bkgid_${id}.json \
#                 --corpus_size 300 --if_use_background_survey ${if_survey} --if_use_strict_survey_question 1 \
#                 --num_screening_window_size 15 --num_screening_keep_size 3 --num_round_of_screening 4 \
#                 --if_save 1 --background_question_id ${id} --if_select_based_on_similarity 0"

#         # Execute the command
#         eval "$full_command"
# done
# echo "Experiment 3 finished successfully"



# ## Experiment 4: Alation Table on strict background

# if_strict_bkg=0
# for id in {0..50}
# do
#         # Construct the full command for each iteration
#         full_command="${base_command}_${model_name_insp_retrieval}_corpusSize_300_survey_1_strict_${if_strict_bkg}_numScreen_15_round_4_similarity_0_bkgid_${id}.json \
#                 --corpus_size 300 --if_use_background_survey 1 --if_use_strict_survey_question ${if_strict_bkg} \
#                 --num_screening_window_size 15 --num_screening_keep_size 3 --num_round_of_screening 4 \
#                 --if_save 1 --background_question_id ${id} --if_select_based_on_similarity 0"

#         # Execute the command
#         eval "$full_command"
# done
# echo "Experiment 4 finished successfully"