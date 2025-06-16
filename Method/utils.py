import os, re, json, random, time, math
import pandas as pd
from google.genai import types


DISCIPLINE = "chemistry"
# MUTATION_CUSTOM_GUIDE: is added to the prompt to mutate to a novel combination (hypothesis) between research background and an inspiration
MUTATION_CUSTOM_GUIDE = "You should be careful on adopting ML methods as the novel content of the mutation, since currently we are using ML examples to illustrate the derivation of hypothesis from research background and inspirations, and now it seems that the ML concepts can therefore easily be abused. "
# HYPTHESIS_GENERATION_CUSTOM_GUIDE: is added to every prompt involving hypothesis generation
HYPTHESIS_GENERATION_CUSTOM_GUIDE = "Please focus on formulating a detailed, valid, feasible, novel, and constructive hypothesis (possibly mainly in terms of the methodology) which will not cause any confusion or misunderstanding on any of its steps: the expected performance or the potential impact of the hypothesis is not the focus and should be mentioned less. "


# A collection of prompts for different modules
# more_info: currently only used in additional_round_inspiration_screening, which is a number indicating the number of inspirations to select
def instruction_prompts(module_name, more_info=None):
    if module_name == "first_round_inspiration_screening":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to an impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Usually a paper can be choosed as an inspiration is because it can potentially help to solve or alleviate one problem of a previous method for this research question so that leveraging the concepts related to the inspiration, a better method can be developed based on the previous methods and this inspiration. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. Here the previous method can only inference the multi-layer logistic regression, but can't automatically update its parameters to learn from data. The selected chain rule inspiration can be leveraged to automatically update the parameters in the multi-layer logistic regression, and therefore improve over the previous method to create hypothesis. \nGiven a research question, the background and some of the existing methods for this research question, and several top-tier publications (including their title and abstract), try to identify which publication can potentially serve as an inspiration for the background research question so that combining the research question and the inspiration in some way, a novel, valid, and significant research hypothesis can be formed. Now try to select inspirations based on the background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe potential inspiration candidates are: ", "\n\nNow you have seen the background research question, existing methods, and many potential inspiration candidates. Please try to identify which three literature candidates are the most possible to serve as the inspiration to the background research question? Please name the title of the literature candidate, and also try to give your reasons. (response format: 'Title: \nReason: \nTitle: \nReason: \nTitle: \nReason: \n')"]
    elif module_name == "first_round_inspiration_screening_only_based_on_semantic_similarity":
        prompts = ["You are helping with the scientists to identify the most semantically similar publications. Given a research question, the background and some of the existing methods for this research question, and several top-tier publications (including their title and abstract), try to identify which publication is the most semantically similar to the background research question. Now try to select publications based on background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe potential publication candidates are: ", "\n\nNow you have seen the background research question, and many potential publication candidates. Please try to identify which three literature candidates are the most semantically similar to the background research question? Please name the title of the literature candidate, and also try to give your reasons. (response format: 'Title: \nReason: \nTitle: \nReason: \nTitle: \nReason: \n')"]
    elif module_name == "additional_round_inspiration_screening":
        # more_info: args.num_screening_keep_size
        assert isinstance(more_info, int)
        assert more_info > 0
        if more_info > 6:
            print(f"Warning: selecting {more_info} inspirations from all inspiration candidates, is it too much?")
        # might choose more than {num_screening_keep_size} inspirations, also might less than {num_screening_keep_size}
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, a core inspiration in a literature for this research question, and a preliminary research hypothesis from the core inspiration. This hypothesis is aiming for top {DISCIPLINE} venue such as <Nature> or <Science>. You know, to publish a research on Nature or Science, the hypotheis must be novel, valid, and significant enough. ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore we also have found a series of inspiration candidates, which might provide additional useful information to assist the core inspiration for the next step of hypothesis generation. We have also obtained the potential hypotheses from the combination of each inspiration candidate with the research background question, which might be helpful in determining how each inspiration candidate can potentially contribute to the research question, and whether it could be helpful / complementary to the preliminary hypothesis developed based on the core inspiration. Please help us select around {more_info} inspiration candidates to assist further development of the hypothesis developed from the core inspiration. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe potential inspiration candidates and their corresponding hypotheses are: ", f"\n\nNow you have seen the background research question, the core inspiration, the preliminary hypothesis, and the potential inspiration candidates with their corresponding hypotheses. Please try to identify which {more_info} inspiration candidates can potentially serve such a complement role for the core inspiration, and how they can be helpful / complementary to the preliminary hypothesis developed based on the core inspiration. (response format: 'Title: \nReason: \nTitle: \nReason: \nTitle: \nReason: \n')"]
    elif module_name == "coarse_hypothesis_generation_only_core_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of conducting research into four steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspiration (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Thirdly it's hypothesis generation based on the background research question and found inspiration; Finally it's about designing and conducting experiments to verify hypothesis. An example is the backpropagation of neural networks. In backpropagation, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. In their paper, the authors have conducted experiments to verify their hypothesis. Now we have identified a good research question, and we have found a core inspiration in a literature for this research question. Please help us generate a novel, valid, and significant research hypothesis based on the background research question and the inspiration. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", f"\n\nNow you have seen the background research question and the core inspiration. Please try to generate a novel, valid, and significant research hypothesis based on the background research question and the inspiration. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    elif module_name == "coarse_hypothesis_generation_without_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of conducting research into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly it's hypothesis generation based on the background research question; Finally it's about designing and conducting experiments to verify hypothesis. An example is the backpropagation of neural networks. In backpropagation, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression, and the research hypothesis is the backpropagation itself. In their paper, the authors have conducted experiments to verify their hypothesis. Now we have identified a good research question. Please help us generate a novel, valid, and significant research hypothesis based on the background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", f"\n\nNow you have seen the background research question. Please try to generate a novel, valid, and significant research hypothesis based on the background research question. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    elif module_name == "validness_checking":
        prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be both novel and valid. Here we focus on the validness aspect. Please try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis by any chance is not valid. If not valid, try to give advice on how it could be modified to be more valid. Please directly answer this question. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis by any chance is not valid. If not valid, what are your advice to be more valid? (response format: 'Yes or No: \nAdvice:\n')"]
    elif module_name == "novelty_checking":
        prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be novel enough, which means it should not have been proposed by any existing literature before. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more novel. If so, what are your advice to be more novel? Please directly answer this question. Please note that your feedback should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more novel. If so, what are your advice to be more novel?"]
    elif module_name == "clarity_checking":
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be clear and specific enough. Please try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific. If so, what are your advice to be more specific? We expect that every detail of the hypothesis is provided, so that a {DISCIPLINE} scientist looking at this hypothesis would have absolutely no doubt on what exactly and comprehensively the hypothesis is in every procedure and in every detail. \
        #            Please directly answer this question. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc. If so, what are your advice to be more specific? (response format: 'Yes or No: \nAdvice:\n')"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a research hypothesis must be clear, specific, and comprehensively detailed.

                    Please carefully evaluate the given hypothesis and answer the following:

                    - Is the hypothesis clear and unambiguous? If not, which parts are vague or confusing?

                    - Does the hypothesis provide sufficient detail on every step, assumption, and condition involved? If not, what additional details would make it more rigorous and specific?

                    - Provide concrete suggestions on how to improve clarity or specificity where needed. When providing the suggestions, you should also ensure that the hypothesis is novel, valid, and significant enough.

                    Your goal is to ensure that any {DISCIPLINE} scientist reading this hypothesis would have absolutely no doubt about its intended meaning, scope, and procedural details.
                   
                    Please directly answer this question. \nThe hypothesis is: \n
                   ''', "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc. If so, what are your advice to be more specific? (response format: 'Yes or No: \nAdvice:\n')"]
    elif module_name == "four_aspects_checking":
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, validness, or significance. \
        #            \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant?"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a strong research hypothesis should satisfy the following four criteria:

                    1. Specificity: The hypothesis should provide sufficient methodological detail so that other researchers can clearly understand what the proposed method is and how it will be carried out in practice, leaving no room for confusion or misinterpretation.

                    In particular, if the hypothesis involves introducing a new concept or component into an existing method, it should not stop at describing what the new concept is — it must also explain how the new concept will be concretely integrated, applied, or operationalized within the method.

                    Whenever possible, please also suggest specific parameters, conditions, or operational procedures (e.g., algorithm settings, material properties, experimental setups) that would enable researchers to directly test or implement the hypothesis in a laboratory or experimental environment.

                    2. Novelty: The hypothesis should propose a new idea, mechanism, or approach that has not been reported or established in existing literature.

                    Please carefully assess whether the core idea of the hypothesis — including its key concepts, methods, or combinations of techniques — has already been proposed or widely studied. If any part of the hypothesis appears similar to prior work, please point it out and explain why it may not be sufficiently novel.

                    Conversely, if the hypothesis is novel, please briefly explain what makes it distinct from existing approaches, such as introducing a new principle, a previously unexplored mechanism, or a new combination of known techniques in an original way.

                    3. Validity / Effectiveness: The hypothesis should be testable, verifiable, and practically feasible within real-world {DISCIPLINE} experimental settings.

                    Please evaluate whether the hypothesis can, in principle, be validated through {DISCIPLINE} experiments — assuming sufficient experimental resources. Consider whether the proposed method relies on reasonable assumptions, whether each step is technically executable, and whether the expected outcomes are measurable in a real-world setting.

                    Although we currently do not have access to lab experiments, please assess the validity based on your knowledge and understanding of {DISCIPLINE}, and highlight any potential challenges, limitations, or conditions that may affect the experimental verification of the hypothesis.

                    4. Significance: If possible, the hypothesis should have the potential for meaningful impact in the research community, such as advancing scientific understanding or opening new research directions. It is not necessary for the hypothesis to be groundbreaking, but it should ideally contribute to the field in a way that is recognized as significant by peers.

                    Please provide constructive feedback on whether the given hypothesis meets these four criteria. If any aspect is lacking, please explain why, and suggest concrete ways to improve it.

                    Important: Your feedback should focus on improving the methodological content of the hypothesis — that is, how to make the hypothesis itself more specific, novel, valid, and significant — rather than suggesting ways to improve the writing or description of these qualities.

                    The hypothesis is: \n
                   ''', "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specific, novel, valid, and significant. If so, what are your advice to be more specific, novel, valid, and significant?"]
    elif module_name == "three_aspects_checking_no_significance":
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; and (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, or valid. If so, what are your advice to be more specific, novel, or valid? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, or validness. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, or valid. If so, what are your advice to be more specific, novel, or valid?"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a strong research hypothesis should satisfy the following four criteria:

                    1. Specificity: The hypothesis should provide sufficient methodological detail so that other researchers can clearly understand what the proposed method is and how it will be carried out in practice, leaving no room for confusion or misinterpretation.

                    In particular, if the hypothesis involves introducing a new concept or component into an existing method, it should not stop at describing what the new concept is — it must also explain how the new concept will be concretely integrated, applied, or operationalized within the method.

                    Whenever possible, please also suggest specific parameters, conditions, or operational procedures (e.g., algorithm settings, material properties, experimental setups) that would enable researchers to directly test or implement the hypothesis in a laboratory or experimental environment.

                    2. Novelty: The hypothesis should propose a new idea, mechanism, or approach that has not been reported or established in existing literature.

                    Please carefully assess whether the core idea of the hypothesis — including its key concepts, methods, or combinations of techniques — has already been proposed or widely studied. If any part of the hypothesis appears similar to prior work, please point it out and explain why it may not be sufficiently novel.

                    Conversely, if the hypothesis is novel, please briefly explain what makes it distinct from existing approaches, such as introducing a new principle, a previously unexplored mechanism, or a new combination of known techniques in an original way.

                    3. Validity / Effectiveness: The hypothesis should be testable, verifiable, and practically feasible within real-world {DISCIPLINE} experimental settings.

                    Please evaluate whether the hypothesis can, in principle, be validated through {DISCIPLINE} experiments — assuming sufficient experimental resources. Consider whether the proposed method relies on reasonable assumptions, whether each step is technically executable, and whether the expected outcomes are measurable in a real-world setting.

                    Although we currently do not have access to lab experiments, please assess the validity based on your knowledge and understanding of {DISCIPLINE}, and highlight any potential challenges, limitations, or conditions that may affect the experimental verification of the hypothesis.

                    Please provide constructive feedback on whether the given hypothesis meets these four criteria. If any aspect is lacking, please explain why, and suggest concrete ways to improve it.

                    Important: Your feedback should focus on improving the methodological content of the hypothesis — that is, how to make the hypothesis itself more specific, novel, and valid — rather than suggesting ways to improve the writing or description of these qualities.

                    The hypothesis is: \n
                   ''', "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specific, novel, and valid. If so, what are your advice to be more specific, novel, and valid?"]
    elif module_name == "four_aspects_checking_and_extra_knowledge":
        prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. In addition, if the hypothesis needs some extra knowledge for it to be more complete, valid, or significant in research, please also try to provide (recall) them (if the hypothesis is already complete, it is not necessary to provide external knowledge). Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, validness, or significance. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? In addition, if the hypothesis need some extra knowledge for it to be more complete, valid, or significant in research, please also try to provide (recall) them."]
    elif module_name == "four_aspects_self_numerical_evaluation":
        # prompts = [f"You are known as a diligent and harsh reviewer in {DISCIPLINE} that will spend much time to find flaws when reviewing and therefore usually gives a relatively much lower score than other reviewers. But when you meet with a hypothesis you truly appreciate, you don't mind to give it good scores. Given a not yet peer reviewed research hypothesis in the {DISCIPLINE} domain, try to evaluate the research hypothesis from four research aspects and give score according to evaluation guidelines provided below. All four aspects should be evaluated in a 5 point scale." + f"\nAspect 1: Validness. \n5 points: The hypothesis is a logical next step from current research, strongly supported by theory, perhaps with some indirect experimental evidence or highly predictive computational results. The experimental verification seems straightforward with a high probability of confirming the hypothesis; 4 points: Here, the hypothesis is well-rooted in existing theory with some preliminary data or computational models supporting it. It extends known science into new but logically consistent areas, where experiments are feasible with current technology, and there's a reasonable expectation of positive results; 3 points: This hypothesis is within the realm of theoretical possibility but stretches the boundaries of what's known. It might combine existing knowledge in very novel ways or predict outcomes for which there's no direct evidence yet. There's a conceptual framework for testing, but success is uncertain; 2 points: While the hypothesis might be grounded in some theoretical aspects, it significantly deviates from current understanding or requires conditions or materials that are currently impossible or highly improbable to achieve or synthesize; 1 point: The hypothesis proposes concepts or outcomes that are not only unsupported by current theory but also contradict well-established principles or data. There's no clear path to experimental testing due to fundamental theoretical or practical barriers. " + f"\nAspect 2: Novelty. \n5 points: This level of novelty could fundamentally alter our understanding of {DISCIPLINE} or create entirely new fields. It often involves predictions or discoveries that, if proven, would require a significant overhaul of existing {DISCIPLINE} theories; 4 points: The hypothesis significantly departs from established norms, potentially redefining how certain {DISCIPLINE} phenomena are understood or applied. It might involve entirely new materials or theoretical frameworks; 3 points: This level involves a hypothesis that could potentially lead to new insights or applications. It might challenge minor aspects of current theories or introduce new methodologies or materials; 2 points: The hypothesis introduces a new angle or method within an established framework. It might involve known compounds or reactions but in contexts or combinations not previously explored; 1 point: The hypothesis involves minor tweaks or applications of well-known principles or techniques. It might slightly extend existing knowledge but doesn't introduce fundamentally new concepts. " + f"\nAspect 3: Significance. \n5 points: This hypothesis could fundamentally change one or more branches of {DISCIPLINE}. It might introduce entirely new principles, theories, or methodologies that redefine the boundaries of {DISCIPLINE}; 4 points: This hypothesis challenges current understanding or introduces a concept that could lead to substantial changes in how a particular area of {DISCIPLINE} is viewed or applied. It might lead to new technologies or significant theoretical advancements; 3 points: this hypothesis proposes something new or an innovative approach that could lead to noticeable advancements in a specific area of {DISCIPLINE}. It might open new avenues for research or application but doesn't revolutionize the field; 2 points: This hypothesis might offer a small variation or incremental improvement on existing knowledge. It could potentially refine a known concept but doesn't significantly alter the field; 1 point: The hypothesis addresses a very narrow or already well-established aspect of {DISCIPLINE}. It might confirm what is already known without adding much new insight." + f"\nAspect 4: Potential. \n5 points: The hypothesis, while potentially intriguing now, holds the promise of being revolutionary with the addition of a key methodological component. This could introduce entirely new concepts or fields, fundamentally changing our understanding or capabilities in {DISCIPLINE}; 4 points: The hypothesis, though promising, could be transformative with the right methodological enhancement. This enhancement might lead to groundbreaking discoveries or applications, significantly advancing the field; 3 points: The hypothesis, while interesting in its current form, could be significantly elevated with the right methodological addition. This might lead to new insights or applications that go beyond the initial scope; 2 points: The hypothesis currently offers some value but has the potential for more substantial contributions if enhanced with a new methodological approach. This could lead to incremental advancements in understanding or application; 1 point: The hypothesis, as it stands, might be straightforward or well-trodden. Even with methodological enhancements, it's unlikely to significantly expand current knowledge or applications beyond minor improvements. \
        #            \nThe hypothesis is:\n", "\nPlease give a response to the initial question on scoring the hypothesis from four aspects. Remember that you are a diligent and harsh reviewer. (response format: 'Concise reason for validness score: \nValidness score: \nConcise reason for novelty score: \nNovelty score: \nConcise reason for significance score: \nSignificance score: \nConcise reason for potential score: \nPotential score: \n')."]
        prompts = [f'''
                    You are a harsh and diligent reviewer in {DISCIPLINE}. You are well-known for carefully identifying flaws and usually giving low scores unless a hypothesis is truly exceptional.
                    Given a not-yet-peer-reviewed research hypothesis in {DISCIPLINE}, evaluate it from four aspects: Validness, Novelty, Significance, and Specificity.
                    Each aspect should be scored from 1 to 5 based on the strict guidelines below. High scores (4 or 5) should only be given to truly outstanding hypotheses. Most ordinary or vague hypotheses should receive scores of 1 to 3. Be strict, objective, and critical. Your evaluation should focus only on the content and methodology of the hypothesis, not on writing style.

                    Scoring Guidelines
                    Validness (Objective Soundness)
                    Does the hypothesis make sense based on your knowledge and reasoning ability?
                    5 — Exceptional soundness. Fully coherent and highly reasonable. No weak assumptions.
                    4 — Strong validity. Mostly reasonable with some uncertain assumptions.
                    3 — Barely valid. Possible but weak, speculative, or fragile reasoning.
                    2 — Low validity. Doubtful or inconsistent.
                    1 — Invalid. Contradicts known science or impossible mechanisms.

                    Novelty (Originality)
                    Is the core idea new?
                    5 — Groundbreaking novelty. Fundamentally new principle or mechanism.
                    4 — Highly novel. Significant deviation from existing knowledge.
                    3 — Moderate novelty. New combination but within existing frameworks.
                    2 — Low novelty. Minor variations of existing work.
                    1 — No novelty. Standard or trivial idea.

                    Significance (Research Impact)
                    If true, how much impact does it have?
                    5 — Field-changing. Reshapes core theories or applications.
                    4 — High impact. Advances state-of-the-art or solves key problem.
                    3 — Meaningful. Improves a subfield or opens new directions.
                    2 — Limited. Incremental improvement or niche value.
                    1 — Minimal. Very narrow or little added value.

                    Specificity (Clarity & Methodological Detail)
                    Is the hypothesis detailed and actionable for scientists?
                    5 — Fully detailed. Every key mechanism and parameter is specified.
                    4 — Detailed. Most steps are clear with minor clarification needed.
                    3 — Moderately clear. General methodology but vague key steps.
                    2 — Low specificity. High-level idea only; hard to use directly.
                    1 — Vague. No actionable detail.
                    The hypothesis is:\n
                   ''', "\nPlease give a response to the initial question on scoring the hypothesis from four aspects. Remember that you are a diligent and harsh reviewer. (response format: 'Concise reason for validness score: \nValidness score: \nConcise reason for novelty score: \nNovelty score: \nConcise reason for significance score: \nSignificance score: \nConcise reason for specificity score: \nSpecificity score: \n')."]
    elif module_name == "hypothesis_generation_with_feedback_only_core_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question and a core inspiration in a literature for this research question. With them, we have already generated a preliminary coarse-grained research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the core inspiration, the preliminary hypothesis, and the feedbacks from domain experts. Please try to refine the hypothesis based on the feedbacks. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nRefined Hypothesis: \n')"]
    elif module_name == "hypothesis_generation_with_feedback_without_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into two steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly it's hypothesis generation based on the background research question. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question. With it, we have already generated a preliminary coarse-grained research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the preliminary hypothesis, and the feedbacks from domain experts. Please try to refine the hypothesis based on the feedbacks. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nRefined Hypothesis: \n')"]
    elif module_name == "hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration":
        # Add "In addition, by generating distinct hypothesis, please do not achieve it by simply introducing new concept(s) into the previous hypothesis to make the difference, but please focus on the difference on the methodology of integrating or leveraging the inspiration to give a better answer to the research question (in terms of the difference on the methodology, concepts can be introduced or deleted)."
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. The experts know that a proper mixture of these components will definitely lead to a valid, novel, and meaningful research hypothesis. In fact, they already have tried to mix them to compose some research hypotheses (that are supposed to be distinct from each other). Please try to explore a new meaningful way to combine the inspiration with the research background to generate a new research hypothesis that is distinct with all the previous hypotheses in terms of their main method. \
                   {MUTATION_CUSTOM_GUIDE} \
                   The new research hypothesis should ideally be novel, valid, ideally significant, and be enough specific in its methodology. Please note that here we are trying to explore a new meaningful way to leverage the inspiration along with the previous methods (inside or outside the introduction) to better answer the background research question, therefore the new research hypothesis should try to leverage or contain the key information or the key reasoning process in the inspiration, trying to better address the background research question. It means the new research hypothesis to be generated should at least not be completely irrelevant to the inspiration or background research question. In addition, by generating distinct hypothesis, please do not achieve it by simply introducing new concept(s) into the previous hypothesis to make the difference, but please focus on the difference on the methodology of integrating or leveraging the inspiration to give a better answer to the research question  (in terms of the difference on the methodology, concepts can be introduced or deleted). \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe previous hypotheses are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and some previous efforts on combining the inspiration with the background for new hypotheses. Please try to generate a novel, valid, detailed, and significant research hypothesis based on the background research question and the inspirations. Please also make sure that the new hypothesis to be generated is distinct with the previous proposed hypotheses in terms of their main method. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    elif module_name == "final_recombinational_mutation_hyp_gene_same_bkg_insp":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. In addition, several experts have already come out of several different hypotheses on how to leverage the inspiration to generate a novel, valid, and significant research hypothesis for the background research question. Please find the bright parts in these hypotheses, leverage the bright parts from them,  modify and combine the good parts of them to generate a better research hypothesis in terms of clarity, novelty, validness, and significance (ideally than any of the given hypotheses). It is not necessary to include methods from every given hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypotheses from different expert teams are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and the hypotheses from different human scientist teams. Please try to generate a novel, valid, significant, and detailed research hypothesis based on the background research question, the inspirations, and the previous efforts from human scientist teams on the given hypotheses. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    elif module_name == "final_recombinational_mutation_hyp_gene_same_bkg_insp_with_feedback":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. In addition, several experts have already come out of several different hypotheses on how to leverage the inspiration to generate a novel, valid, and significant research hypothesis for the background research question. Please find the bright parts in these hypotheses, leverage the bright parts from them,  modify and combine the good parts of them to generate a better research hypothesis in terms of clarity, novelty, validness, and significance (ideally than any of the given hypotheses). It is not necessary to include methods from every given hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given hypothesis. In fact, a researcher has already tried to propose hypothesis based on these information, and we have obtained the feedback to his hypothesis, from another respectful researcher. Please try to leverage the feedback to improve the hypothesis, you can leverage all these provided information as your reference. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypotheses from different expert teams are: ", "\n\nThe hypothesis from the researcher is: ", "\n\nThe feedback to the hypothesis from the researcher is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypotheses from different human scientist teams, the hypothesis from the researcher, and the feedback to the hypothesis from the researcher. Please try to generate a better hypothesis (in terms of novelty, validness, significance, and detailedness) based on these information. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nRefined Hypothesis: \n')"]
    elif module_name == "final_recombinational_mutation_hyp_gene_between_diff_inspiration":
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, a core inspiration in a literature for this research question, and a hypothesis resulted from leveraging the core inspiration to answer the research background question. This hypothesis is aiming for top {DISCIPLINE} venues such as <Nature> or <Science>. You know, to publish a research on <Nature> or <Science>, the hypotheis must be novel, valid, and significant enough. Ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore a senior researcher have identified an additional inspiration, along with a hypothesis generated from leveraging the additional inspiration to the research background question. This additional inspiration and its corresponding hypothesis is supposed to provide complementry useful information to assist the further development of the hypothesis developed from the core inspiration. Please find the bright parts in these hypotheses, try to leverage the bright parts from them, modify the hypothesis developed based on the given core inspiration to improve it in terms of novelty, validness, significance, and detailedness. It is not necessary to include methods from every given inspiration & its hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given inspiration & its hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe hypotheses from other inspirations are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, and the hypotheses resulted from different inspirations. Please try to generate a better hypothesis (in terms of novelty, validness, significance, and detailedness) based on these information. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    elif module_name == "final_recombinational_mutation_hyp_gene_between_diff_inspiration_with_feedback":
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. \nNow we have identified a good research question, an introduction of previous methods, a core inspiration in a literature for this research question, and a hypothesis resulted from leveraging the core inspiration to answer the research background question. This hypothesis is aiming for top {DISCIPLINE} venues such as <Nature> or <Science>. You know, to publish a research on <Nature> or <Science>, the hypotheis must be novel, valid, and significant enough. Ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore a senior researcher have identified an additional inspiration, along with a hypothesis generated from leveraging the additional inspiration to the research background question. This additional inspiration and its corresponding hypothesis is supposed to provide complementry useful information to assist the further development of the hypothesis developed from the core inspiration. Please find the bright parts in these hypotheses, try to leverage the bright parts from them, modify the hypothesis developed based on the given core inspiration to improve it in terms of novelty, validness, significance, and detailedness. In fact, a researcher has already tried to propose hypothesis based on these information, and we have obtained the feedback to his hypothesis, from another respectful researcher. Please try to leverage the feedback to improve the hypothesis, you can leverage all these provided information as your reference. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe hypotheses from other inspirations are: ", "\n\nThe hypothesis from the researcher is: ", "\n\nThe feedback to the hypothesis from the researcher is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, the hypotheses resulted from different inspirations, the hypothesis from the researcher, and the feedback to the hypothesis from the researcher. Please try to generate a better hypothesis (in terms of novelty, validness, significance, and detailedness) based on these information. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nRefined Hypothesis: \n')"]
    elif module_name == "self_extra_knowledge_exploration":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Please try your best to explore one more knowledge that can potentially improve or complement the existing research hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and the hypothesis from the core inspiration, please try to explore one more knowledge that can potentially improve or complement the existing research hypothesis. If the hypothesis is complete enough, please answer 'No' to 'If need extra knowledge:' template, and give your reason in 'Details' template. If extra knowledge is needed, please answer 'Yes' first, and then to give the explored knowledge in 'Details' template. (response format: 'If need extra knowledge: \nDetails: \n')"]
    elif module_name == "self_extra_knowledge_exploration_with_other_mutations":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Please try your best to explore one more knowledge that can potentially improve or complement the existing research hypothesis. One more thing to mention, the researchers have already tried to further develop the original hypothesis with extra knowledge, and they have already proposed some potential hypotheses afterwards. Here we want to explore the extra knowledge in a different way with these hypotheses. So please try to develop the original hypothesis with extra knowledge, but not in the same way as any of the hypothesis developed afterwards, so to explore more opportunities. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe original hypothesis from the core inspiration is: ", "\n\nThe hypotheses developed afterwards are: ", "\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the original hypothesis from the core inspiration, and some hypotheses developed afterwards based on the original hypothesis, please try to explore one more knowledge that can potentially improve or complement the original research hypothesis, but not in the same way as any of the hypothesis developed afterwards. If the original hypothesis is complete enough, please answer 'No' to 'If need extra knowledge:' template, and give your reason in 'Details' template. If extra knowledge is needed, please answer 'Yes' first, and then to give the explored knowledge in 'Details' template. (response format: 'If need extra knowledge: \nDetails: \n')"]
    elif module_name == "hypothesis_generation_with_extra_knowledge":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Therefore the researcher has already explored the additional knowledge to make the hypothesis more complete. Please try your best to generate a new hypothesis based on the background research question, the inspiration, the additional knowledge, and the given preliminary hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe additional knowledge is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, and the additional knowledge, please try to generate a new hypothesis based on the background research question, the inspiration, the hypothesis, and the additional knowledge. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nHypothesis: \n')"]
    # here with_extra_knowledge" means the hypothesis is generated based on the core inspiration and the extra knowledge, but not that the feedback need to cover extra knowledge
    elif module_name == "provide_feedback_to_hypothesis_four_aspects_with_extra_knowledge":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generate a hypothesis for the research question based on the little survey, an extra knowledge that should be usedful to develop a hypothesis, and the hypotheses developed based on the inspiration and the extra knowledge. Please try to give some feedback to the research hypothesis. Specifically, you know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding (if it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab); (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. \nPlease try your best to give the senior researcher some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, significance, or validness. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe extra knowledge is: ", "\n\nThe hypothesis is: ", "\n\nNow you have seen the background research question, the core inspiration, the extra knowledge, and the hypothesis. Please give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant?"]
    elif module_name == "hypothesis_refinement_with_feedback_with_extra_knowledge":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into four steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Thirdly it's about finding extra knowledge that work along with the inspiration can lead to a more complete hypothesis. Finally it's hypothesis generation based on the background research question, the found inspirations, and the extra knowledge. \nNow we have identified a good research question, a core inspiration in a literature for this research question, and extra knowledge. With them, we have already generated a preliminary research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe extra knowledge is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the core inspiration, the extra knowledge, the preliminary hypothesis, and the feedbacks from domain experts. Please try to refine the hypothesis based on the feedbacks. {HYPTHESIS_GENERATION_CUSTOM_GUIDE}(response format: 'Reasoning Process:\nRefined Hypothesis: \n')"]
    elif module_name == "eval_matched_score":
        prompts = [f"You are helping to evaluate the quality of a proposed research hypothesis in {DISCIPLINE} by a phd student. The groundtruth hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key points in terms of the methodology in the groundtruth hypothesis. You will also be given a summary of the key points in the methodology of the groundtruth hypothesis for reference. Please note that for the proposed hypothesis to cover one key point, it is not necessary to explicitly mention the name of the key point, but might also can integrate the key point implicitly in the proposed method. The evaluation criteria is called 'Matched score', which is in a 6-point Likert scale (from 5 to 0). Particularly, 5 points mean that the proposed hypothesis (1) covers all the key points and leverage them similarly as in the methodology of the groundtruth hypothesis, and (2) does not contain any extra key point that has apparent flaws; 4 points mean that the proposed hypothesis (1) covers all the key points (or at least three key points) and leverage them similarly as in the methodology of the groundtruth hypothesis, (2) but also with extra key points that have apparent flaws; 3 points mean that the proposed hypothesis (1) covers at least two key points and leverage them similarly as in the methodology of the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, (3) might or might not contain extra key points; 2 points mean that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth hypothesis, and leverage it similarly as in the methodology of groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; 1 point means that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth hypothesis, (2) but is used differently as in the methodology of groundtruth hypothesis, and (3) might or might not contain extra key points; 0 point means that the proposed hypothesis does not cover any key point in the methodology of the groundtruth hypothesis at all. Please note that the total number of key points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there's only one key point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point, it's possible to give 2 points, 4 points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality of extra key points. 'Leveraging a key point similarly as in the methodology of the groundtruth hypothesis' means that in the proposed hypothesis, the same (or very related) concept (key point) is used in a similar way with a similar goal compared to the groundtruth hypothesis (not necessarily for the proposed hypothesis to be exactly the same with the groudtruth hypothesis to be classified as 'similar'). When judging whether an extra key point has apparent flaws, you should use your own knowledge to judge, but rather than to rely on the count number of pieces of extra key point to judge. \nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis. \nThe proposed hypothesis is: ", "\n\nThe groundtruth hypothesis is: ", "\n\nThe key points in the groundtruth hypothesis are: ", "\n\nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis, and give a score. (response format: 'Reason: \nMatched score: \n')"]
    elif module_name == "eval_matched_score_hard":
        prompts = ["You are helping to evaluate the quality of a proposed research hypothesis by a phd student. The groundtruth hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key points of the groundtruth hypothesis. You will also be given a summary of the key points in the groundtruth hypothesis for reference. The evaluation criteria is called 'Matched score', which is in a 6-point Likert scale (from 5 to 0). Particularly, \n5 points mean that the proposed hypothesis (1) covers three key points (or covers all the key points) in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, and (2) does not contain any extra key point(s) that is redundant, unnecessary, unhelpful, or harmful; \n4 points mean that the proposed hypothesis (1) covers three key points (or covers all the key points) in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, and (2) but also contain extra key point(s) that is redundant, unnecessary, unhelpful, or harmful; \n3 points mean that the proposed hypothesis (1) covers two key points in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; \n2 points mean that the proposed hypothesis (1) covers one key point in the groundtruth hypothesis, and leverage it nearly identically as in the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; \n1 point means that the proposed hypothesis (1) covers at least one key point in the groundtruth hypothesis, but all the covered key point(s) are used differently as in the groundtruth hypothesis, and (2) might or might not contain extra key points; \n0 point means that the proposed hypothesis does not cover any key point in the groundtruth hypothesis at all. \nUsually total the number of key points a groundtruth hypothesis contain is less than or equal to three. Please note that the total number of key points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there's only one key point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point nearly identically, it's possible to give 2 points, 4 points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality of extra key points. 'Leveraging a key point nearly identically as in the groundtruth hypothesis means that in the proposed hypothesis, the same (or very related) concept (key point) is used in a very similar way with a very similar goal compared to the groundtruth hypothesis. \nWhen judging whether an extra key point has apparent flaws, you should use your own knowledge and understanding of that discipline to judge, rather than only relying on the count number of pieces of extra key point to judge. \nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis. \nThe proposed hypothesis is: ", "\n\nThe groundtruth hypothesis is: ", "\n\nThe key points in the groundtruth hypothesis are: ", "\n\nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis, and give a score. (response format: 'Reason: \nMatched score: \n')"]
    else:
        raise NotImplementedError
    
    return prompts


# Input:
#   input_list: [[item0, item1], [item0, item1], ...] OR [item0, item1]
# Output:
#   output_list: [[item1, item0], [item1, item0], ...] OR [item1, item0]
#   The order of the items in the input_list is reversed.
#   If the input_list is a list of lists, the order of the items in each list is reversed.
#   If the input_list is a list of strings, the order of the items in the list is reversed.
def exchange_order_in_list(input_list):
    output_list = []
    for cur_input_list in input_list:
        if isinstance(cur_input_list, list):
            assert len(cur_input_list) == 2
            output_list.append(cur_input_list[::-1])
        elif isinstance(cur_input_list, str):
            assert len(input_list) == 2
            output_list = input_list[::-1]
            break
        else:
            raise ValueError("Invalid input type. Expected list or string.")
    return output_list



# calculate the ratio if how the selected inspirations hit the groundtruth inspirations. 
def calculate_average_ratio_top1_top2(file_dir):
    with open(file_dir, 'r') as f:
        d = json.load(f)

    ratio_top1, ratio_top2 = 0, 0
    cnt_ratio = 0
    for i in d[1]:
        cur_ratio = d[1][i]
        ratio_top1 += cur_ratio[0]
        ratio_top2 += cur_ratio[1]
        cnt_ratio += 1
    ratio_top1 = ratio_top1 / cnt_ratio
    ratio_top2 = ratio_top2 / cnt_ratio
    return ratio_top1, ratio_top2


## Function: used by load_chem_annotation() and load_chem_annotation_with_feedback(); used to recover background_survey_strict and background_question_strict
# background_strict_raw: a list of the raw background survey, some of them are "NA"; when it is "NA", we should find its component in background_normal
# background_normal: a list of the normal background survey, no "NA"
# background_strict_raw_nan_indicator: a list of boolean values indicating whether the corresponding background_strict_raw is "NA"
def recover_raw_background(background_strict_raw, background_normal, background_strict_raw_nan_indicator):
    background_strict = []
    for cur_survey_id, cur_survey in enumerate(background_strict_raw):
        if background_strict_raw_nan_indicator[cur_survey_id]:
            cur_value = background_normal[cur_survey_id].strip()
            background_strict.append(cur_value)
        else:
            cur_survey = cur_survey.strip()
            # this assertion is to make sure the content is not variants of "NA"
            assert len(cur_survey) > 10
            cur_value = cur_survey
            background_strict.append(cur_value)
    return background_strict
    


# load xlsx annotations, bkg question -> inspirations
# bkg_q: [bq0, bq1, ...]
# dict_bkg2insp: {'bq0': [insp0, insp1, ...], 'bq1': [insp0, insp1, ...], ...}
# dict_bkg2survey: {'bq0': survey0, 'bq1': survey1, ...}
def load_chem_annotation(chem_annotation_path, if_use_strict_survey_question, if_use_background_survey=1):
    assert if_use_strict_survey_question in [0, 1]
    assert if_use_background_survey in [0, 1]
    if if_use_background_survey == 0:
        print("Warning: Not Using Survey.")
    ## load chem_research.xlsx to know the groundtruth inspirations
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    nan_values = chem_annotation.isna()
    bkg_survey = list(chem_annotation[chem_annotation.columns[4]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_survey
    bkg_survey_strict_raw = list(chem_annotation[chem_annotation.columns[5]])
    # print("bkg_survey_strict_raw: ", bkg_survey_strict_raw)
    bkg_survey_strict = recover_raw_background(bkg_survey_strict_raw, bkg_survey, nan_values[chem_annotation.columns[5]])
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_q
    bkg_q_strict_raw = list(chem_annotation[chem_annotation.columns[7]])
    bkg_q_strict = recover_raw_background(bkg_q_strict_raw, bkg_q, nan_values[chem_annotation.columns[7]])
    insp1 = list(chem_annotation[chem_annotation.columns[9]])
    insp2 = list(chem_annotation[chem_annotation.columns[11]])
    insp3 = list(chem_annotation[chem_annotation.columns[13]])
    groundtruthHyp = list(chem_annotation[chem_annotation.columns[15]])
    reasoningprocess = list(chem_annotation[chem_annotation.columns[17]])
    note = list(chem_annotation[chem_annotation.columns[18]])
    ## determine which version of survey and question to use
    if if_use_strict_survey_question:
        bkg_survey = bkg_survey_strict
        bkg_q = bkg_q_strict
    ## start looping for collection
    dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2reasoningprocess = {}, {}, {}, {}, {}
    dict_bkg2idx, dict_idx2bkg = {}, {}
    for cur_b_id, cur_b in enumerate(bkg_q):
        # update bkg_q to remove leading and trailing spaces
        cur_b = cur_b.strip()
        bkg_q[cur_b_id] = cur_b
        ## dict_bkg2insp
        cur_b_insp = []
        # insp1
        if nan_values[chem_annotation.columns[9]][cur_b_id] == False:
            cur_b_insp.append(insp1[cur_b_id].strip())
        # insp2
        if nan_values[chem_annotation.columns[11]][cur_b_id] == False:
            cur_b_insp.append(insp2[cur_b_id].strip())
        # insp3
        if nan_values[chem_annotation.columns[13]][cur_b_id] == False:
            cur_b_insp.append(insp3[cur_b_id].strip())
        dict_bkg2insp[cur_b] = cur_b_insp
        ## dict_bkg2survey
        if if_use_background_survey:
            assert nan_values[chem_annotation.columns[4]][cur_b_id] == False
            dict_bkg2survey[cur_b] = bkg_survey[cur_b_id].strip()
        else:
            dict_bkg2survey[cur_b] = "Survey not provided. Please overlook the survey."
        ## dict_bkg2groundtruthHyp
        assert nan_values[chem_annotation.columns[15]][cur_b_id] == False
        dict_bkg2groundtruthHyp[cur_b] = groundtruthHyp[cur_b_id].strip()
        ## dict_bkg2reasoningprocess
        assert nan_values[chem_annotation.columns[17]][cur_b_id] == False
        dict_bkg2reasoningprocess[cur_b] = reasoningprocess[cur_b_id].strip()
        ## dict_bkg2note
        assert nan_values[chem_annotation.columns[18]][cur_b_id] == False
        dict_bkg2note[cur_b] = note[cur_b_id].strip()
        ## dict_bkg2idx, dict_idx2bkg
        dict_bkg2idx[cur_b] = cur_b_id
        dict_idx2bkg[cur_b_id] = cur_b
    return bkg_q, dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2idx, dict_idx2bkg, dict_bkg2reasoningprocess


# load xlsx annotations and data id, return the background question and inspirations; used for check_moosechem_output() in analysis.py
def load_bkg_and_insp_from_chem_annotation(chem_annotation_path, background_question_id, if_use_strict_survey_question):
    # load chem_research.xlsx to know the groundtruth inspirations
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    nan_values = chem_annotation.isna()
    # bkg_survey = list(chem_annotation[chem_annotation.columns[4]])
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    bkg_q_strict_raw = list(chem_annotation[chem_annotation.columns[7]])
    bkg_q_strict = recover_raw_background(bkg_q_strict_raw, bkg_q, nan_values[chem_annotation.columns[7]])
    insp1 = list(chem_annotation[chem_annotation.columns[9]])
    insp2 = list(chem_annotation[chem_annotation.columns[11]])
    insp3 = list(chem_annotation[chem_annotation.columns[13]])
    # whether use strict version of bkg_q
    if if_use_strict_survey_question:
        bkg_q = bkg_q_strict

    cur_bkg = bkg_q[background_question_id].strip()
    cur_insp_list = []
    # insp1
    if nan_values[chem_annotation.columns[9]][background_question_id] == False:
        cur_insp_list.append(insp1[background_question_id].strip())
    # insp2
    if nan_values[chem_annotation.columns[11]][background_question_id] == False:
        cur_insp_list.append(insp2[background_question_id].strip())
    # insp3
    if nan_values[chem_annotation.columns[13]][background_question_id] == False:
        cur_insp_list.append(insp3[background_question_id].strip())
    return cur_bkg, cur_insp_list

    

# load the title and abstract of the groundtruth inspiration papers and random high-quality papers
# INPUT
#   title_abstract_collector_path: the file path of the inspiration corpus
#       It should contain a list of [title, abstract] pairs: [[title, abstract], ...]
# OUTPUT
#   title_abstract_collector: [[title, abstract], ...]
#   dict_title_2_abstract: {'title': 'abstract', ...}
def load_dict_title_2_abstract(title_abstract_collector_path):
    ## load title_abstract_collector
    with open(title_abstract_collector_path, 'r') as f:
        # title_abstract_collector: [[title, abstract], ...]
        title_abstract_collector = json.load(f)
    print("Number of title-abstract pairs loaded: ", len(title_abstract_collector))
    ## Transfer title_abstract_collector to dict_title_2_abstract
    # dict_title_2_abstract: {'title': 'abstract', ...}
    dict_title_2_abstract = {}
    for cur_item in title_abstract_collector:
        if cur_item[0] in dict_title_2_abstract:
            # print("Warning: seen before: ", cur_item[0])
            continue
        dict_title_2_abstract[cur_item[0]] = cur_item[1]
    return title_abstract_collector, dict_title_2_abstract


# inspiration_path: path to selected inspiration, eg, "coarse_inspiration_search_gpt4.json"
# load coarse-grained / fine-grained inspiration screening results
## Output
# organized_insp: {'bq': [[title, reason], [title, reason], ...]}
def load_found_inspirations(inspiration_path, idx_round_of_first_step_insp_screening):
    with open(inspiration_path, 'r') as f:
        selected_insp_info = json.load(f)
    # organized_insp: {'bq': [screen_results_round1, screen_results_round2, ...], ...}
    #   screen_results_round1: [[title, reason], [title, reason], ...]
    organized_insp = selected_insp_info[0]
    organized_insp_hit_ratio = selected_insp_info[1]
    # dict_bkg_insp2idx: {'bq': {'title': idx, ...}, ...}
    # dict_bkg_idx2insp: {'bq': {idx: 'title', ...}, ...}
    dict_bkg_insp2idx, dict_bkg_idx2insp = {}, {}
    # organized_insp_selected_round: {'bq': [[title, reason], [title, reason], ...]}
    organized_insp_selected_round = {}
    for bq in organized_insp:
        dict_bkg_insp2idx[bq] = {}
        dict_bkg_idx2insp[bq] = {}
        organized_insp_selected_round[bq] = []
        for idx, cur_insp in enumerate(organized_insp[bq][idx_round_of_first_step_insp_screening]):
            dict_bkg_insp2idx[bq][cur_insp[0]] = idx
            dict_bkg_idx2insp[bq][idx] = cur_insp[0]
            organized_insp_selected_round[bq].append(cur_insp)
        print("\nNumber of inspirations loaded: {} for background question: {}".format(len(organized_insp_selected_round[bq]), bq))
    return organized_insp_selected_round, dict_bkg_insp2idx, dict_bkg_idx2insp


## Input
# bkg_q: text
# dict_bkg2insp: {'bq0': [insp0, insp1, ...], 'bq1': [insp0, insp1, ...], ...}
## Output
# organized_insp: {'bq': [[title, reason], [title, reason], ...]}
# dict_bkg_insp2idx: {'bq': {'title': idx, ...}, ...}
# dict_bkg_idx2insp: {'bq': {idx: 'title', ...}, ...}
def load_groundtruth_inspirations_as_screened_inspirations(bkg_q, dict_bkg2insp):
    # organized_insp
    organized_insp = {}
    organized_insp[bkg_q] = []
    # dict_bkg_insp2idx, dict_bkg_idx2insp
    dict_bkg_insp2idx, dict_bkg_idx2insp = {}, {}
    dict_bkg_insp2idx[bkg_q] = {}
    dict_bkg_idx2insp[bkg_q] = {}
    # iterating through the inspirations
    gdth_insps = dict_bkg2insp[bkg_q]
    for cur_insp_id, cur_insp in enumerate(gdth_insps):
        organized_insp[bkg_q].append([cur_insp, "Not provided yet."])
        dict_bkg_insp2idx[bkg_q][cur_insp] = cur_insp_id
        dict_bkg_idx2insp[bkg_q][cur_insp_id] = cur_insp
    return organized_insp, dict_bkg_insp2idx, dict_bkg_idx2insp



## Input
# selected_insp: {'bq': [screen_results_round1, screen_results_round2, ...], ...}
#   screen_results_round1: [[[title, reason], [title, reason]], [[title, reason], [title, reason]], ...]
## Output
# organized_insp: {'bq': [screen_results_round1_org, screen_results_round2_org, ...]}
#   screen_results_round1_org: [[title, reason], [title, reason], ...]
def organize_raw_inspirations(selected_insp):
    # organized_insp: {'bq': [[title, reason], [title, reason], ...]}
    organized_insp = {}
    for bq in selected_insp:
        assert bq not in organized_insp
        organized_insp[bq] = []
        # cur_screen_results_round: [[[title, reason], [title, reason]], [[title, reason], [title, reason]], ...]
        for cur_round_id, cur_screen_results_round in enumerate(selected_insp[bq]):
            organized_insp[bq].append([])
            # round_insp: [[title, reason], [title, reason]] (most likely only two or three inspirations)
            for round_insp in cur_screen_results_round:
                organized_insp[bq][cur_round_id] += round_insp
    return organized_insp


# insp_grouping_results: {insp title: [[other insp title, reason], ...]}
def load_grouped_inspirations(inspiration_group_path):
    with open(inspiration_group_path, 'r') as f:
        insp_grouping_results = json.load(f)
    return insp_grouping_results


# coarse_grained_hypotheses: {core_insp_title: [[hypothesis, reasoning process], ...]}
def load_coarse_grained_hypotheses(coarse_grained_hypotheses_path):
    with open(coarse_grained_hypotheses_path, 'r') as f:
        coarse_grained_hypotheses = json.load(f)
    return coarse_grained_hypotheses
    

# Call Openai API,k input is prompt, output is response
# model: by default is gpt3.5, can also use gpt4
def llm_generation(prompt, model_name, client, temperature=1.0, api_type=0):
    # which model to use
    if_api_completed = False
    # start inference util we get generation
    while if_api_completed == False:
        # print("api_type: ", api_type)   
        try:
            if api_type in [0, 1]:
                completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ]
                )
                generation = completion.choices[0].message.content
            elif api_type == 2:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                generation = response.text
            else:
                raise NotImplementedError
            if_api_completed = True
        except Exception as e:
            print("OpenAI reaches its rate limit: ", e)
            time.sleep(0.25)
    return generation


## Function:
#   llm inference with the prompt + guarantee to reply a structured generation accroding to the template (guarantee by the while loop)
#   gene_format_constraint: [id of structured gene to comply with the constraint, constraint (['Yes', 'No'], where the content in the id of structured gene should be inside the constraint)]
#   if_only_return_one_structured_gene_component: True or False; most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]]). When it is True, this function will only return the first element of structured_gene. If it is set to true and structured_gene has more than one component, a warning will be raised
def llm_generation_while_loop(prompt, model_name, client, if_structured_generation=False, template=None, gene_format_constraint=None, if_only_return_one_structured_gene_component=False, temperature=1.0, restructure_output_model_name=None, api_type=0):
    # assertions
    assert if_structured_generation in [True, False]
    if if_structured_generation:
        assert template is not None
    if restructure_output_model_name == None:
        restructure_output_model_name = model_name
    else:
        if restructure_output_model_name != model_name:
            print(f"Warning: restructure_output_model_name is set to {restructure_output_model_name}, which is different from model_name: {model_name}.")

    # while loop to make sure there will be one successful generation
    while True:
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # structured_gene
            if if_structured_generation:
                # structured_gene: [[title, reason], [title, reason], ...]
                # try with template matching first, if not work, try llm to formulate the generation according to the template; if not work again, then probably it is the problem of the original generation, then try llm_generation() again
                try:
                    # print("Using a template matching method to extract information from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
                except:
                    # print("Information to be extracted by an LLM from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation_by_llm(generation, template=template, client=client, temperature=temperature, model_name=restructure_output_model_name, api_type=api_type)
                if gene_format_constraint != None:
                    assert len(gene_format_constraint) == 2, print("gene_format_constraint: ", gene_format_constraint)
                    # we use structured_gene[0] here since most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]])
                    assert structured_gene[0][gene_format_constraint[0]].strip() in gene_format_constraint[1], print("structured_gene[0][gene_format_constraint[0]].strip(): {}; gene_format_constraint[1]: {}".format(structured_gene[0][gene_format_constraint[0]].strip(), gene_format_constraint[1]))
            break
        except Exception as e:
            # if the format of feedback is wrong, try again in the while loop
            # print("generation: ", generation)
            print("AssertionError: {}, try again..".format(repr(e)))
            

    # structured_gene
    if if_structured_generation:
        if if_only_return_one_structured_gene_component:
            if len(structured_gene) > 1:
                print("Warning: structured_gene has more than one component: ", structured_gene)
            return structured_gene[0]
        else:
            return structured_gene
    else:
        return generation



def get_structured_generation_from_raw_generation_by_llm(gene, template, client, temperature, model_name, api_type=0):
    assert isinstance(gene, str), print("type(gene): ", type(gene))
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # In your answer, please only mention the words in the template when use it as a template. For example, if the template is ['Hypothesis:', 'Reasoning Process:'], then your answer should not contain 'Analysis of the Hypothesis:', since it also contain 'Hypothesis:'.
    # Whenever there are information in the passage related to the template, please restructure the information into the template format;
    prompt = "You are a helpful assistant.\nPlease help to organize the following passage into a structured format, following the template. When restructure the passage with the template, please try not to rephrase but to use the original information in the passage (to avoid information distortion). If the template is only about a subset of information in the passage, you can extract only that subset of information to fill the template. If there is no such information for the template in the passage, please still output the exact template first, and fill the content for the template as 'None'. \n\nThe passage is: \n" + gene + f"\n\nThe template is: \n{template[0]} \n{template[1]} \n. Now, please restructure the passage strictly with the template (literally strictly, e.g., the case style of the template should also remain the same when used to restructure the passage)."
    # print("prompt: ", prompt)
    
    # while loop to make sure there will be one successful generation
    max_trials = 50
    for cur_trial in range(max_trials):
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # print("generation (in): ", generation)
            structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
            # print("structured_gene (in): ", structured_gene)
            return structured_gene
        except Exception as e:
            if temperature < 2.0:
                temperature += 0.25
            # if temperature >= 0.7:
            #     model_name = "gpt-4o"
            # if the format of feedback is wrong, try again in the while loop
            print("generation (in): ", generation)
            print("template: ", template)
            print("Exception (in): {}, try again..".format(repr(e)))
            print(f"update temperature to {temperature} and use {model_name} for extraction in case new generation can be successful..")
    # print("structured_gene: ", structured_gene)
    raise Exception("Failed to restructure the passage with the template after {} trials.".format(max_trials))





# gene: (generated) text; '#' and '*' will be removed from gene, since they are assumed to be generated by LLM as markdown format --- this format can result in not exact match between the title extracted from generation and the groundtruth title in the benchmark
# template: ['Title:', 'Reason:']
# structured_gene: [[Title, Reason], ...]
def get_structured_generation_from_raw_generation(gene, template):
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # # some times generation will capitalize the first letter of the template, so we use the lower case for both generation and template to match: not adopting it since it might influence chemistry terms (e.g., Fe2+ -> fe2+)
    # gene = gene.lower()
    # template = [item.lower() for item in template]
    # the template might not appear in the first sentence of the gene, get rid of noise sentences before the first template[0]
    if not gene.startswith(template[0]):
        gene_split = gene.split('\n')
        # if the gene is not starting with the title, the second paragraph in gene_split might be the title
        gene_split = [item for item in gene_split if item.strip() != ""]
        assert len(gene_split) >= 2, print("gene_split: ", gene_split)
        # iterate to find the first template[0] in gene_split
        for id_line, line in enumerate(gene_split):
            if gene_split[id_line].find(template[0]) > 0 and gene_split[id_line].find(template[0]) < 15:
                gene_split_split = gene_split[id_line].split(template[0])
                assert len(gene_split_split) == 2, print("gene_split_split: ", gene_split_split)
                gene_split[id_line] = template[0] + gene_split_split[1]
            if gene_split[id_line].startswith(template[0]):
                gene = '\n'.join(gene_split[id_line:])
                break
        # assert gene.startswith(template[0]), print("gene: ", gene)
        assert gene.startswith(template[0])
    # structured_gene: [[title, reason], [title, reason], ...]
    structured_gene = []
    gene_split = gene.split(template[0])
    # split to every title block, including one title and one reason
    for cur_gs in gene_split:
        # split the one title and one reason
        cur_gs = cur_gs.strip()
        if cur_gs == "":
            continue
        cur_gs_split = cur_gs.split(template[1])
        # deal with unexpected situations
        if len(cur_gs_split) > 2:
            # if there are more than one template[1] in cur_gs, we prefer the one with prefix as '\n' (since it matches more as the designed format)
            cur_gs_split = cur_gs.split('\n' + template[1])
            # by preferring the one with prefix as '\n' still do not work, so we simply concatenate the rest of the elements other than the first element
            if len(cur_gs_split) > 2:
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
            # in case none of the template[1] is with prefix as '\n'
            elif len(cur_gs_split) == 1:
                cur_gs_split = cur_gs.split(template[1])
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
        # assert len(cur_gs_split) == 2, print("cur_gs_split: ", cur_gs_split)
        assert len(cur_gs_split) == 2
        # strip every elements in cur_gs_split
        for i in range(len(cur_gs_split)):
            cur_gs_split[i] = cur_gs_split[i].strip().strip(";").strip()
        structured_gene.append(cur_gs_split)
    return structured_gene



def pick_score(cur_generation):
    """
    We have 4 categories, each with two parts:
      (1) Concise reason for X score:
          -> reason text might be on the same line or the next line
      (2) X score:
          -> numeric score might be on the same line or the next line

    For example, any of the following is valid:

        Concise reason for validness score:
        <REASON>

        Validness score: 4

    or

        Concise reason for validness score: The reason is ...
        Validness score: 4

    or even:

        Concise reason for validness score:
        The reason is ...
        Validness score:
        4

    This function tolerates extra text before or after the structure.
    """

    reason_labels = [
        "Concise reason for validness score:",
        "Concise reason for novelty score:",
        "Concise reason for significance score:",
        "Concise reason for specificity score:"
    ]
    score_labels = [
        "Validness score:",
        "Novelty score:",
        "Significance score:",
        "Specificity score:"
    ]

    # Acceptable numeric scores
    valid_scores = {"1", "2", "3", "4", "5"}

    # Break into lines
    raw_lines = cur_generation.split('\n')
    # Strip trailing/leading whitespace from each line (but keep line breaks for logic)
    lines = [line.strip() for line in raw_lines]

    # For final results
    reason_collection = []
    score_collection = []

    # We'll parse in the order: reason_label -> reason_text -> score_label -> numeric_score,
    # repeated 4 times (for validness, novelty, significance, potential).

    # We keep a pointer as we move through lines
    i = 0
    n = len(lines)

    # Function to find the next line (starting at index start_idx) that *contains* `label`.
    # Returns the index (line number) or None if not found.
    def find_label(label, start_idx):
        label_lower = label.lower()
        for idx in range(start_idx, n):
            if lines[idx].lower().startswith(label_lower):
                return idx
        return None

    # Function to extract content either on the same line (after the label) OR from the next line(s).
    #  - If there's text following the label on the same line, use that.
    #  - Otherwise, look at the next line if it exists.
    # Returns (content_str, next_idx), or (None, None) on failure.
    def extract_content(label, label_idx):
        # The line with the label:
        line_with_label = lines[label_idx]

        # Check if the label and content are on the same line
        # e.g. "Concise reason for validness score: The reason is right here"
        # We'll remove the label part from that line and see if there's leftover text
        stripped_label_part = line_with_label[: len(label)]
        leftover = line_with_label[len(label):].strip(" \t:")  # remove trailing : and spaces

        if leftover:
            # We have content on the same line
            return leftover, label_idx + 1
        else:
            # No leftover text on the same line; we want the next line if it exists
            if label_idx + 1 < n:
                # The next line is presumably the content
                next_line_content = lines[label_idx + 1].strip()
                if next_line_content:
                    return next_line_content, label_idx + 2
                else:
                    # The next line was empty, so maybe we look further?
                    # You could decide to skip blank lines. Here let's skip blank lines
                    j = label_idx + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n:
                        return lines[j].strip(), j + 1
                    else:
                        return None, None
            else:
                return None, None

    # Now parse each of the four reason+score pairs
    search_start = 0
    for reason_label, score_label in zip(reason_labels, score_labels):

        # 1) Find the reason label
        pos_reason_label = find_label(reason_label, search_start)
        if pos_reason_label is None:
            print(f"Could not find label: '{reason_label}' starting at line {search_start}.")
            return [], [], False

        # 2) Extract reason text
        reason_text, next_idx = extract_content(reason_label, pos_reason_label)
        if reason_text is None:
            print(f"No reason text found after label '{reason_label}'.")
            return [], [], False
        reason_collection.append(reason_text)

        # 3) Find the corresponding score label, starting at next_idx
        pos_score_label = find_label(score_label, next_idx)
        if pos_score_label is None:
            print(f"Could not find label: '{score_label}' after '{reason_label}'.")
            return [], [], False

        # 4) Extract numeric score
        score_text, next_idx_score = extract_content(score_label, pos_score_label)
        if score_text is None:
            print(f"No numeric score found after label '{score_label}'.")
            return [], [], False

        # Clean up the score text: remove 'points', 'point', '*' if present
        clean_score = (
            score_text.replace("points", "")
                      .replace("point", "")
                      .replace("*", "")
                      .strip()
        )
        if clean_score not in valid_scores:
            print(f"Score '{clean_score}' is not in {valid_scores}.")
            return [], [], False
        score_collection.append(int(clean_score))

        # Update our search position so that the next search starts after we used the score
        search_start = next_idx_score

    # Finally, we consider it successful if we have 4 reasons and 4 scores
    if len(reason_collection) == 4 and len(score_collection) == 4:
        return score_collection, reason_collection, True
    else:
        print("Mismatch in number of scores/reasons extracted.")
        print("cur_generation:", cur_generation)
        return [], [], False


## Function
#  calculate the average score of the four aspects. The score range is [0, 1]
def jaccard_similarity(str1, str2):
    words1 = set(str1.split())
    words2 = set(str2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)



# some titles are generated by LLM, which might have slight different from the exact title extracted from the markdown file
# groundtruth_titles: [title, ...], extracted from markdown file
# title: title generated by LLM 
def title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles, if_print_warning=True):
    assert if_print_warning in [True, False]
    # groundtruth_titles:  [title, ...]
    similarity_collector = []
    for cur_item in groundtruth_titles:
        cur_similarity = jaccard_similarity(title.lower(), cur_item.lower()) 
        similarity_collector.append(cur_similarity)
    # get the most similar one
    max_similarity = max(similarity_collector)
    max_similarity_index = similarity_collector.index(max_similarity)
    matched_title = groundtruth_titles[max_similarity_index]
    if max_similarity < 0.3 and if_print_warning:
        print("max_similarity: {}; original title: {}; \nmatched title: {}\n".format(max_similarity, title, matched_title))
    return matched_title, max_similarity


# dict_title_2_abstract: a dict with groundtruth title as key, and abstract as value
# groundtruth_titles: [title, ...], extracted from markdown file
# title: title generated by LLM, that might not be exactly the same as the groundtruth title key in dict_title_2_abstract
## Output
# value: the abstract corresponding to the title
def get_item_from_dict_with_very_similar_but_not_exact_key(dict_title_2_abstract, title):
    groundtruth_titles = list(dict_title_2_abstract.keys())
    try:
        value = dict_title_2_abstract[title]
    except:
        title, similarity = title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles)
        value = dict_title_2_abstract[title]
    return value


## Function:
#   generated title might be different from the exact title in the groundtruth title list, this function is to recover the generated title to the exact version of the title in the groundtruth title list
# groundtruth_titles: [title, ...]
# title: title generated by LLM
def recover_generated_title_to_exact_version_of_title(groundtruth_titles, title):
    title = title.strip().strip('"').strip()
    recovered_title, similarity = title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles)
    return recovered_title


## Function:
#   whether an element is in a list with a similarity threshold (if th element has a similarity larger than the threshold with any element in the list, return True)
def if_element_in_list_with_similarity_threshold(list_elements, element, threshold=0.7):
    element = element.strip().strip('"').strip()

    for cur_element in list_elements:
        cur_element = cur_element.strip().strip('"').strip()
        if jaccard_similarity(element.lower(), cur_element.lower()) > threshold:
            return True
    return False


def save_with_json(data, file_dir):
    with open(file_dir, 'w') as f:
        json.dump(data, f)



# Function: transfer a list to set, while maintaining order
def ordered_set(input_list):
    set_list = []
    for item in input_list:
        if item not in set_list:
            set_list.append(item)
    return set_list

    





