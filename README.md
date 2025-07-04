# MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses



<!-- <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FZonglinY%2FMOOSE-Chem&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a> -->
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Us)](https://x.com/Yang_zy223)
[![GitHub Repo stars](https://img.shields.io/github/stars/ZonglinY/MOOSE-Chem%20)](https://github.com/ZonglinY/MOOSE-Chem)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.07076)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

We introduce **MOOSE-Chem**, which is an LLM-based multi-agent framework for automated chemistry scientific hypothesis discovery. 

With only LLMs with training data up to October 2023, it has rediscovered many chemistry hypotheses published on Nature, Science, or similar levels in 2024 (also only available online in 2024) with very high similarity, covering the main innovations.



<p align="center" width="100%">
  <img src="./Resources/main_figure_io_insp_corpus.png" alt="MOOSE-Chem" style="width: 75%; display: block; margin: auto;"></a>
</p>

The input to MOOSE-Chem can be as simple as only:

&emsp;(1) *research question*: can be on any chemistry & material science domain;

&emsp;(2) *background survey*: (optionally) a several-paragraph-long survey describing the existing methods for the *research question*;

&emsp;(3) *inspiration corpus*: (this repo contains the default 3000 papers) title and abstract of many (random) chemistry papers that might serve as inspirations for the *research question*, preferably published on top venues.

**MOOSE-Chem** can then output a list of ranked chemistry hypotheses (might take a few hours to "think") that could be both novel and valid.


---------- 

This repo contains all the code of **MOOSE-Chem**, to help every chemistry lab to catalyze their chemistry scientific discovery process.

In general, **MOOSE-Chem** contains three stages:  
&emsp;(1) inspiration retrieval;  
&emsp;(2) hypothesis composition;   
&emsp;(3) hypothesis ranking.

The commands for the three stages are introduced after the "quick start".

--- 

## ⚡ Step 0: Quick Start

```bash
git clone https://github.com/ZonglinY/MOOSE-Chem.git
cd MOOSE-Chem
conda create -n msc python=3.10
conda activate msc
pip install -r requirements.txt
```

Then, open `main.sh` and configure the following parameters:

* `api_type`
* `api_key`
* `base_url`
* `model_name_insp_retrieval`
* `model_name_gene`
* `model_name_eval`

> 🔧 **Note:**
> Set `api_type` to `0` if you're using an OpenAI API key, and to `1` if you're using an Azure OpenAI API key.
>
> 💡 **Tip:**
> You can assign the same model name to all three tasks (`model_name_insp_retrieval`, `model_name_gene`, and `model_name_eval`).

---

## 📋 Step 1: (Optional) Provide Custom Research Background — or Use the Default Benchmark

You can supply your own `research_question` and `background_survey` as input. Otherwise, the system will use a built-in benchmark.

#### To provide custom input:

1. Open `./Preprocessing/custom_research_background_dumping_and_output_displaying.py`
2. In the `research_background_to_json()` function, manually fill in:
   - `research_question`
   - `background_survey`
3. In `main.sh`:
   - Set `custom_research_background_path`, which stores the customized `research_question` and `background_survey`
   - Comment out all other Python commands **except** the one below

```bash
## Custom Research Background Dumping
## Modify 'research_question' and 'background_survey' in research_background_to_json()
python -u ./Preprocessing/custom_research_background_dumping_and_output_displaying.py --io_type 0 \
    --custom_research_background_path ${custom_research_background_path}
```

Then run `bash main.sh`.

> ✅ Once done, this will generate a custom research background file that can be used in later steps.

---



## 📋 Step 2: (Optional) Use a Custom Inspiration Corpus — or Stick with the Default

You can provide your own inspiration corpus (titles and abstracts) to set up the hypothesis search space. If not provided, the system will use the default ones in the benchmark dataset.

### ✅ Two Ways to Provide a Custom Corpus:

#### **Option 1**: Manually Compose Your Own

Prepare a list of papers and save them in the following format:

```python
[[title1, abstract1], [title2, abstract2], ...]
```

Save this to the path specified by `custom_inspiration_corpus_path`.

---

#### **Option 2**: Use Web of Science to Download in Bulk

1. **Prepare the Raw Data**
   * Use [Web of Science](https://www.webofscience.com/wos/woscc/summary/0d1f66e0-aebb-4b29-a6c8-d685e04c2ea9-015bae6080/relevance/1) to search for papers by journal name and optionally filter with keywords.
   * Select the desired papers by checking their boxes.
   * Click **"Export"** in the top menu.
   * Choose **"Excel"** as the format.
   * Set **"Record Content"** to **"Author, Title, Source, Abstract"**
   * Click **"Export"** to download the file (The file should have a `.xlsx` or `.xls` extension).

   Save all `.xlsx` or `.xls` files into a single folder for further processing.


2. **Edit `main.sh`**  
   - Set `custom_raw_inspiration_data_dir` to the folder path containing your `.xlsx` or `.xls` files  
   - Set `custom_inspiration_corpus_path` to the desired output path for the processed inspiration corpus  
   - Comment out all other Python commands **except** the one below

```bash
## Custom Inspiration Corpus Construction
python -u ./Preprocessing/construct_custom_inspiration_corpus.py \
    --raw_data_dir ${custom_raw_inspiration_data_dir} \
    --custom_inspiration_corpus_path ${custom_inspiration_corpus_path}
```

3. **Run the Script**

   ```bash
   bash main.sh
   ```

> ✅ Once done, this will generate a custom inspiration corpus file that can be used in later steps.


---

## 🔍 Step 3: Inspiration Retrieval

This step retrieves relevant literature snippets to serve as inspiration for hypothesis generation.

#### If you're using custom research background and inspiration corpus:

The following variables in `main.sh` should have been set up in Step 1 and Step 2 (just keep them as is):

- `custom_research_background_path`  
- `custom_inspiration_corpus_path`

#### If you're using the default benchmark:

Set both variables to empty strings:

```bash
custom_research_background_path=""
custom_inspiration_corpus_path=""
```

#### In `main.sh`:

Comment out all other Python commands **except** the one below:

```bash
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
        --if_save 1 --background_question_id 0 --if_select_based_on_similarity 0 \
        --custom_research_background_path ${custom_research_background_path} \
        --custom_inspiration_corpus_path ${custom_inspiration_corpus_path}
```

Then run `bash main.sh`.

<!-- Customized *research question* and *background survey* can be used by modifying ```custom_rq, custom_bs = None, None``` to any string in inspiration_screening.py. -->

<!-- Customized *inspiration corpus* can be adopted by setting ```--title_abstract_all_insp_literature_path``` to your customized file with format ```[[title, abstract], ...]```. -->

---

## 🧠 Step 4: Hypothesis Composition

#### In `main.sh`:

1. Keep the `custom_research_background_path` and `custom_inspiration_corpus_path` as which you have selected in your previous step.

2. Comment out all other Python commands **except** the one below:

```bash
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
```

Here ```--inspiration_dir``` should be set the same as ```--output_dir``` used in the inspiration retrieval step.

<!-- Customized *research question* and *background survey* can be used by modifying ```custom_rq, custom_bs = None, None``` to any string in hypothesis_generation.py. -->

<!-- If used customized *inspiration corpus* in the inspiration retrieval step, ```--title_abstract_all_insp_literature_path``` should be set as the same file as used in the inspiration retrieval step. -->

Then run `bash main.sh`.

---

## 📈 Step 5: Hypothesis Ranking

#### In `main.sh`:

1. Keep the `custom_inspiration_corpus_path` as which you have selected in your previous step.

2. Comment out all other Python commands **except** the one below:

```bash
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
```

Here ```--hypothesis_dir``` should be set the same as ```-output_dir``` used in the hypothesis composition step.

If used customized *research question* and *background survey*, ```--if_with_gdth_hyp_annotation``` should be set to 0, unless the groundtruth hypothesis can be obtained for the customized *research question* (in this case the function to load groundtruth hypothesis ```load_chem_annotation()``` need to be modified).

<!-- If used customized *inspiration corpus* in the inspiration retrieval and hypothesis composition steps, ```--title_abstract_all_insp_literature_path``` should be set as the same file as used in these steps.   -->

Then run `bash main.sh`.

---

## 🖥️ Step 6: (Optional) Display Ranked Generated Hypothesis

#### In `main.sh`:

1. Set up `display_txt_file_path` to a txt file where displays the ranked generated hypothesis.

2. Comment out all other Python commands **except** the one below:

```bash
## Hypothesis Display
python -u ./Preprocessing/custom_research_background_dumping_and_output_displaying.py --io_type 1 \
        --evaluate_output_dir ${checkpoint_root_dir}/evaluation_${model_name_eval}_${output_dir_postfix}.json \
        --display_dir ${display_txt_file_path}
```

Then run `bash main.sh`.

---


<!-- These basic commands for the three stages can also be found in ```main.sh```.  -->

```Assumption1-RetrieveUnseen.sh``` and ```Assumption2-Reason2Unknown.sh``` contain combinations of these three basic commands (with different arg parameters) to investigate LLMs' ability on these three aspects.

---

## Analysis

```./Analysis/analysis.py``` can be used to analyze the results of the three stages. 
This [link](https://drive.google.com/file/d/1WdnB5Ztb4n3DNfwJeE9GJW-BJvdoWmNN/view?usp=sharing) stores the result files from all the experiments mentioned in the paper. They can be used with ```analysis.py``` to display the experiment results reported in the paper.

---

## An Example

Here we present a rediscovered hypothesis from MOOSE-Chem, with input:

(1) a research question && a survey on existing methods for the question; and

(2) 300 random chemistry papers published on Nature or Science, containing two groundtruth inspirations papers.

### Ground Truth Hypothesis

*The main hypothesis is that a **nitrogen-doped ruthenium (Ru)** electrode can effectively catalyze the reductive deuteration of (hetero)arenes in the presence of **D_2O**, leading to high deuterium incorporation into the resulting saturated cyclic compounds. The findings validate this hypothesis by demonstrating that this electrocatalytic method is highly efficient, scalable, and versatile, suitable for a wide range of substrates.*

### Rediscovered Hypothesis

*A pioneering integrated electrocatalytic system leveraging **ruthenium** nanoparticles embedded in **nitrogen-doped** graphene, combined with a dual palladium-coated ion-exchange membrane reactor, will catalyze efficient, scalable, and site-selective reductive deuteration of aromatic hydrocarbons and heteroarenes. Utilizing deuterium sources from both $D_2$ gas and **D_2O**, this system will optimize parameters through real-time machine learning-driven dynamic adjustments. Specific configurations include ruthenium nanoparticle sizes (2-4 nm), nitrogen doping levels (12-14\%), precisely engineered palladium membranes (5 micrometers, ensuring 98\% deuterium-selective permeability), and advanced cyclic voltammetry protocols (1-5 Hz, -0.5V to -1.5V).*

### Expert's analysis 

The proposed hypothesis effectively covers two key points from the ground truth hypothesis: **the incorporation of ruthenium (Ru) and the use of D_2O as a deuterium source** within the electrocatalytic system. However, the current content does not detail the mechanism by which Ru-D is produced, which is essential for explaining the process of reductive deuteration. Nevertheless, the results are still insightful. The specific level of nitrogen doping, for example, is highly suggestive and warrants further investigation. Overall, the match remains strong in its alignment with the original hypothesis while also presenting opportunities for deeper exploration.

---

## Bib Info
If you found this repository useful, please consider 📑citing:

	@inproceedings{yang2024moose,
	  title={MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses},
	  author={Yang, Zonglin and Liu, Wanhao and Gao, Ben and Xie, Tong and Li, Yuqiang and Ouyang, Wanli and Poria, Soujanya and Cambria, Erik and Zhou, Dongzhan},
	  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
	  year={2025}
	}


---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
